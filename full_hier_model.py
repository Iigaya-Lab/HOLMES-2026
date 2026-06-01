

"""
HOLMES with sticky nested Chinese Restaurant Process (nCRP)

 particle filter for hierarchical causal inference:
- Multi-particle hierarchical CRP prior with depth-decay
- Canonical global node registry for tree structure sharing
- Depth-decayed persistence mechanism for path reuse
- Persistence bonus kappa_l is tied to omega and decays with depth
- Optional prediction-error attenuation weakens persistence after surprising outcomes
- Partial observation/feedback support (for complex experiments)
- Bernoulli feature likelihoods (computed in log-space to avoid underflow)
- Tree-aware online reward prediction: reward estimates are mixed across
  all nodes on the sampled path, so parent/ancestor statistics can support
  generalization before leaf-specific evidence is available.
"""

import numpy as np


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_hist(labels, weights, K):
    """
    Compute weighted histogram of labels.
    
    Args:
        labels: Array of integer labels
        weights: Corresponding weights for each label
        K: Number of bins (0 to K-1)
    
    Returns:
        Array of shape (K,) with weighted counts
    """
    mass = np.zeros(K, dtype=float)
    for lab, w in zip(labels, weights):
        lab = int(lab)
        if 0 <= lab < K:
            mass[lab] += w
    return mass


def _bernoulli_predictive(obs_bool_vec, n_vec, b_vec, eps=1e-12):
    """
    Compute Bernoulli predictive probabilities.
    
    For each feature:
    - If observed (obs=1): p = n/(n+b)
    - If absent (obs=0): p = b/(n+b)
    
    Args:
        obs_bool_vec: Boolean observation vector
        n_vec: Counts of feature being present
        b_vec: Counts of feature being absent
        eps: Small constant for numerical stability
    
    Returns:
        Predictive probability for each feature
    """
    denom = np.maximum(n_vec + b_vec, eps)
    p_present = n_vec / denom
    p_absent = b_vec / denom
    O = obs_bool_vec.astype(np.float64)
    return O * p_present + (1.0 - O) * p_absent


def _log_likelihood_sum(
    obs_bool,
    nFC,
    bFC,
    cause_k,
    part_l,
    feat_idx,
    length_normalize=False,
    tau=1.0,
    obs_mask=None,
    eps=1e-12
):
    """
    Compute sum of log feature likelihoods for a given cause.
    
    Args:
        obs_bool: Boolean observation vector (nFeatures,)
        nFC: Feature-present counts (nFeatures, nMaxCauses, nParticles)
        bFC: Feature-absent counts (nFeatures, nMaxCauses, nParticles)
        cause_k: Cause index
        part_l: Particle index
        feat_idx: Indices of features to include in likelihood
        length_normalize: If True, use geometric mean (average log-likelihood)
        tau: Temperature parameter for length normalization
        obs_mask: Boolean mask of observed features
        eps: Small constant for numerical stability
    
    Returns:
        Log-likelihood value (sum or mean of log-likelihoods)
    """
    if obs_mask is None:
        obs_mask = np.ones_like(obs_bool, dtype=bool)

    # Restrict to observed and selected features
    effective_idx = [i for i in feat_idx if obs_mask[i]]
    if len(effective_idx) == 0:
        return 0.0  # log(1) = 0

    p_f = _bernoulli_predictive(
        obs_bool[effective_idx],
        nFC[effective_idx, cause_k, part_l],
        bFC[effective_idx, cause_k, part_l],
        eps=eps
    )

    # Compute log-likelihoods
    log_p_f = np.log(np.maximum(p_f, eps))
    
    if length_normalize:
        # Geometric mean with temperature (mean log-likelihood)
        return float(tau * np.mean(log_p_f))
    else:
        # Sum of log-likelihoods (= log of product)
        return float(np.sum(log_p_f))


# ============================================================
# GLOBAL NODE REGISTRY
# ============================================================

class GlobalNodeRegistry:
    """
    Shared canonical structure ensuring that identical signatures
    across particles map to the same global node ID.
    
    Implements node reuse as in the nCRP:
    - Identical (level, parent_id, branch_index) → same global node
    - Maintains parent-child adjacency for tree exploration
    """
    
    def __init__(self, max_children=5):
        """
        Args:
            max_children: Maximum number of children per node
        """
        self.max_children = max_children
        self.registry = {}   # (level, parent_id, branch_idx) -> global_id
        self.children = {}   # parent_id -> [child_global_ids]
        self.next_id = 1

        # Root node always has ID = 0
        self.registry[(0, None, 0)] = 0
        self.children[None] = [0]
        self.children[0] = []

    def get_or_create(self, level, parent_id, branch_index):
        """
        Get existing node ID or create new one for given signature.
        
        Args:
            level: Depth in tree
            parent_id: Global ID of parent node
            branch_index: Branch index at this level
        
        Returns:
            Global node ID
        """
        sig = (level, parent_id, branch_index)
        if sig in self.registry:
            gid = self.registry[sig]
        else:
            gid = self.next_id
            self.registry[sig] = gid
            self.next_id += 1

            # Register this new child
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(gid)
            self.children[gid] = []

        return gid

    def get_children(self, parent_id):
        """Get list of child node IDs for given parent."""
        return self.children.get(parent_id, [])


# ============================================================
# PER-PARTICLE nCRP TREE
# ============================================================

class NCRPTreeParticle:
    """Stores per-particle, parent-specific CRP counts.

    In a true nCRP, the restaurant at level L is conditional on the
    currently selected parent node. Therefore branch counts cannot be
    stored only by level. They must be stored by (level, parent_gid):

        child_counts[(L, parent_gid)] = [count_child_0, count_child_1, ...]

    This lets different parents at the same depth maintain different child
    distributions, which is the defining nested part of the nCRP.
    """

    def __init__(self, alpha, max_depth, max_children):
        """
        Args:
            alpha: Base concentration parameter
            max_depth: Maximum tree depth
            max_children: Maximum branches per parent node
        """
        self.alpha = float(alpha)
        self.max_depth = int(max_depth)
        self.max_children = max_children
        # Parent-specific branch counts. Key: (level, parent_gid).
        # Value: list of counts for child branch indices under that parent.
        self.child_counts = {}


# ============================================================
# HIERARCHICAL CRP PRIOR
# ============================================================

class MHLCMCRP:
    """
    Multi-particle nCRP prior.
    
    Features:
    - Depth-decayed CRP concentration parameter
    - Canonical global node registry with reuse across particles
    - Per-particle leaf mapping to cause IDs
    - Stickiness mechanism favoring reuse of previous paths
    - Stopping probability that decreases with depth
    """

    def __init__(
        self,
        alpha,
        max_depth,
        nParticles,
        nMaxCauses,
        random_seed=None,
        depth_decay=None,
        max_children=5,
        stickiness=1.0,
        stickiness_decay=None,
        pe_adapt_stickiness=True,
        pe_sensitivity=2.0
    ):
        """
        Args:
            alpha: Base CRP concentration parameter
            max_depth: Maximum tree depth
            nParticles: Number of particles
            nMaxCauses: Maximum number of causes
            random_seed: Random seed for reproducibility
            depth_decay: Rate of decay for concentration with depth. If None,
                uses alpha, implementing the bounded-rationality coupling
                alpha_l = alpha * exp(-alpha * level).
            max_children: Maximum number of children per node
            stickiness: Base persistence strength. In the HOLMES paper logic this
                is tied to omega. The level-specific value is
                kappa_l = stickiness * exp(-stickiness_decay * level).
            stickiness_decay: Depth decay for kappa_l. If None, uses stickiness,
                giving the analogous omega-tied schedule
                kappa_l = omega * exp(-omega * level).
            pe_adapt_stickiness: If True, previous-trial prediction error
                attenuates persistence on the next trial.
            pe_sensitivity: Strength of prediction-error attenuation.
        """
        self.alpha = float(alpha)
        self.max_depth = int(max_depth)
        self.nParticles = nParticles
        self.nMaxCauses = nMaxCauses
        # Keep alpha tied to itself by default, as in the paper:
        # alpha_l = alpha * exp(-alpha * level).
        self.depth_decay = float(alpha if depth_decay is None else depth_decay)
        self.max_children = max_children

        # Base persistence is tied to omega by the caller. By default, its
        # decay is tied to itself too: kappa_l = omega * exp(-omega * level).
        self.stickiness = float(stickiness)
        self.stickiness_decay = float(stickiness if stickiness_decay is None else stickiness_decay)
        self.pe_adapt_stickiness = bool(pe_adapt_stickiness)
        self.pe_sensitivity = float(pe_sensitivity)

        self.rng = np.random.RandomState(random_seed)

        # Per-particle CRP trees
        self.trees = [
            NCRPTreeParticle(alpha, max_depth, max_children)
            for _ in range(nParticles)
        ]

        # GLOBAL leaf to cause mapping (shared across all particles!)
        self.leaf_map = dict()  # global_node_id -> cause_id
        self.leaf_next = 0  # next available cause ID
        
        # Global structure
        self.global_registry = GlobalNodeRegistry(max_children=max_children)
        self.level_gid_sets = [set() for _ in range(max_depth)]
        
        # Previous paths for persistence
        self.prev_path = {m: None for m in range(nParticles)}

        # Previous-trial prediction error per particle. This is used only to
        # attenuate persistence on the next trial; it does not encode any
        # task-specific state.
        self.prev_prediction_error = np.zeros(nParticles, dtype=float)

    def resample(self, idx):
        """
        Resample particles according to indices.
        
        Args:
            idx: Array of particle indices to keep
        """
        self.trees = [self.trees[i] for i in idx]
        self.prev_path = {m: self.prev_path[i] for m, i in enumerate(idx)}
        self.prev_prediction_error = self.prev_prediction_error[idx].copy()
        # Note: leaf_map and leaf_next are now global so no resampling

    def set_prediction_error(self, prediction_error):
        """Store previous-trial prediction error for adaptive persistence.

        The next call to generate_prior can use these errors to reduce the
        persistence bonus after surprising outcomes. This remains task-general:
        surprise weakens the prior to stay in the same path, but does not specify
        which level or state should change.
        """
        pe = np.asarray(prediction_error, dtype=float)
        if pe.shape != (self.nParticles,):
            raise ValueError(
                f"prediction_error must have shape ({self.nParticles},), got {pe.shape}"
            )
        self.prev_prediction_error = np.clip(pe, 0.0, 1.0)

    def _alpha_level(self, level):
        """Depth-decayed concentration, alpha_l = alpha * exp(-alpha * level) by default."""
        return float(self.alpha * np.exp(-self.depth_decay * level))

    def _kappa_level(self, level, particle_idx):
        """Depth-decayed persistence bonus tied to omega/stickiness.

        kappa_l = kappa_0 * exp(-rho * level), with kappa_0 normally equal
        to omega and rho normally equal to omega. If enabled, previous-trial
        prediction error attenuates this bonus on the next trial.
        """
        kappa = self.stickiness * np.exp(-self.stickiness_decay * level)
        if self.pe_adapt_stickiness:
            pe = self.prev_prediction_error[particle_idx]
            kappa *= np.exp(-self.pe_sensitivity * pe)
        return float(kappa)

    def generate_prior(self, return_paths=False):
        """
        Generate hierarchical CRP paths for all particles.
        - sample from the prior and then weight by likelihood.
        
        Args:
            return_paths: If True, return both assignments and path arrays
        
        Returns:
            assignments: Cause ID for each particle (nParticles,)
            paths_global: (Optional) Global node IDs along each path (nParticles, max_depth)
        """
        assignments = np.zeros(self.nParticles, dtype=int)
        paths_global = -np.ones((self.nParticles, self.max_depth), dtype=int)
        # Reset level tracking
        self.level_gid_sets = [set() for _ in range(self.max_depth)]

        for m in range(self.nParticles):
            particle = self.trees[m]
            parent_gid = 0
            path_gids = []
            prev_path = self.prev_path.get(m, None)

            for L in range(self.max_depth):
                # Depth-decayed concentration. With depth_decay=None in __init__,
                # this is alpha_L = alpha * exp(-alpha * L), preserving the
                # bounded-rationality coupling described in the paper.
                alpha_L = self._alpha_level(L)

                # Stopping process after depth L, so depth 0 always continues.
                if L > 0:
                    stop_prob = 1.0 / (1.0 + alpha_L)
                    if self.rng.rand() < stop_prob:
                        break

                # True nested-CRP bookkeeping: counts are conditional on
                # the current parent node, not shared globally across all nodes
                # at the same depth. Each (level, parent_gid) pair has its own
                # local restaurant over child branches.
                count_key = (L, parent_gid)
                counts = particle.child_counts.setdefault(count_key, [])
                K = len(counts)

                # Compute branch probabilities
                if K == 0:
                    # No existing children under this parent: must create first one
                    probs = np.array([1.0])
                    choice = 0
                    counts.append(1)
                else:
                    # Parent-specific CRP probabilities
                    if K < self.max_children:
                        # Can create new child branch under this parent
                        probs = np.zeros(K + 1)
                        probs[:K] = np.array(counts, dtype=float)
                        probs[K] = alpha_L
                    else:
                        # At capacity: choose existing children only
                        probs = np.array(counts, dtype=float)

                    # Apply additive, depth-decayed persistence bonus.
                    # This is the nCRP analogue of sticky HDP-HMM persistence:
                    #   P(branch k) ∝ n_k + kappa_l * 1[k = previous branch]
                    # and P(new) ∝ alpha_l.
                    # kappa_l is tied to omega/stickiness and decays with depth,
                    # so high-level structure persists longer without predefining
                    # the number or identity of levels.
                    if (prev_path is not None and
                        L < len(prev_path) and
                        prev_path[L] >= 0):
                        prev_gid = prev_path[L]
                        kappa_L = self._kappa_level(L, m)
                        # Find which existing branch matches previous path.
                        for existing in range(K):
                            candidate_gid = self.global_registry.get_or_create(
                                L, parent_gid, existing
                            )
                            if candidate_gid == prev_gid:
                                probs[existing] += kappa_L
                                break

                    # Normalize probabilities
                    probs_norm = probs / np.sum(probs)
                    
                    # Sample branch
                    choice = self.rng.choice(len(probs_norm), p=probs_norm)
                    
                    # Update counts
                    if choice == K and K < self.max_children:
                        counts.append(1)
                    else:
                        counts[choice] += 1

                # Get or create global node ID
                gid = self.global_registry.get_or_create(L, parent_gid, choice)
                parent_gid = gid
                path_gids.append(gid)
                self.level_gid_sets[L].add(gid)

            # Store path
            full_path = -np.ones(self.max_depth, dtype=int)
            full_path[:len(path_gids)] = path_gids
            paths_global[m] = full_path

            # Map leaf to cause ID 
            if len(path_gids) == 0:
                # Empty path: use root as leaf
                leaf_gid = 0
            else:
                leaf_gid = path_gids[-1]
                
            # Check global leaf map (shared across all particles)
            if leaf_gid not in self.leaf_map:
                leaf_id = self.leaf_next
                if leaf_id >= self.nMaxCauses:
                    raise RuntimeError(
                        f"Exceeded nMaxCauses={self.nMaxCauses} at global leaf {leaf_gid}"
                    )
                self.leaf_map[leaf_gid] = leaf_id
                self.leaf_next += 1

            assignments[m] = self.leaf_map[leaf_gid]

        # Store paths for next iteration (stickiness)
        self.prev_path = {m: paths_global[m].copy() for m in range(self.nParticles)}

        if return_paths:
            return assignments, paths_global
        return assignments


# ============================================================
# FULL INFERENCE LOOP
# ============================================================

def full_hier_inference_loop(
    nTrials,
    nParticles,
    nFeatures,
    alpha,
    omega,
    f,
    max_depth=10,
    max_children=5,
    random_seed=None,
    outcome_idx=2,
    outcome_idx_per_trial=None,
    include_outcome_in_weight=True,
    length_normalize=False,
    tau=1.0,
    feedback_mask=None,
    return_rprob=False,
    use_tree_prediction=True,
    tree_mix=0.5,
    uncertainty_k=10.0,
    level_decay=1.0,
    stickiness_decay=None,
    pe_adapt_stickiness=True,
    pe_sensitivity=2.0,
    prediction_prior_mix=0.5,
    path_readout_mode="most_informative"
):
    """
    Run HOLMES!!!!
    
    Args:
        nTrials: Number of trials
        nParticles: Number of particles
        nFeatures: Number of features per observation
        alpha: CRP concentration parameter
        omega: Prior pseudocount (aPrior = bPrior = omega) and stickiness
        f: Observation matrix (nFeatures, nTrials)
        max_depth: Maximum tree depth
        max_children: Maximum branches per node
        random_seed: Random seed
        outcome_idx: Index of outcome feature
        outcome_idx_per_trial: Per-trial outcome indices (optional)
        include_outcome_in_weight: Whether to include outcome in particle weights
        length_normalize: Use geometric mean for likelihoods
        tau: Temperature for length normalization
        feedback_mask: Observation mask (None, (nTrials,), or (nFeatures, nTrials))
    
    Returns:
        particles: Cause assignments (nParticles, nTrials)
        cEst: Posterior over causes (nMaxCauses, nTrials)
        rEst: Reward predictions (nTrials,)
        phiEst: Feature expectations (nFeatures, nTrials)
        cEst_levels: Posteriors at each tree level (max_depth, nMaxCauses, nTrials)
        paths_dense: Reindexed paths (nParticles, nTrials, max_depth)
    """
    
    # ============================================================
    # SETUP
    # ============================================================
    
    rng = np.random.RandomState(random_seed)
    aPrior = bPrior = omega
    nMaxCauses = nTrials*max_depth
    eps = 1e-12

    # Particle state
    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + aPrior
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + bPrior

    # Additional feature/outcome counts for every tree node, not just leaves.
    # These are the minimal additions that let online reward prediction exploit
    # the full path. A parent node can accumulate evidence across many leaves,
    # supporting transfer to a new or sparsely observed child.
    node_nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + aPrior
    node_bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + bPrior

    # Weights and predictions
    wt = np.ones(nParticles) / nParticles
    rt = np.ones(nParticles) / nParticles
    rProb = np.zeros(nParticles)
    rProb_parts= np.zeros((nParticles, nTrials))

    # Estimates
    cEst = np.zeros((nMaxCauses, nTrials))
    rEst = np.zeros(nTrials)
    phiEst = np.zeros((nFeatures, nTrials))
    paths = -np.ones((nParticles, nTrials, max_depth), dtype=int)
    cEst_levels = np.zeros((max_depth, nMaxCauses, nTrials))

    all_idx = np.arange(nFeatures)

    # ============================================================
    # FEEDBACK MASK PROCESSING
    # ============================================================
    
    if feedback_mask is None:
        # Full feedback on all trials
        use_fb = np.ones(nTrials, dtype=bool)
        feat_mask = np.ones((nFeatures, nTrials), dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask)

        if feedback_mask.ndim == 1:
            # Trial-level mask (e.g., mushroom task)
            use_fb = feedback_mask.astype(bool)
            feat_mask = np.tile(use_fb, (nFeatures, 1))
        elif feedback_mask.ndim == 2:
            # Feature-level mask (e.g., conditioning task)
            feat_mask = feedback_mask.astype(bool)
            use_fb = np.any(feat_mask, axis=0)
        else:
            raise ValueError("feedback_mask must be None, (T,) or (nFeatures, T)")

    # ============================================================
    # INITIALIZE PRIOR MODEL
    # ============================================================
    
    path_readout_mode = str(path_readout_mode).lower()
    valid_path_readout_modes = {"weighted", "most_informative", "max_confidence", "winner"}
    if path_readout_mode not in valid_path_readout_modes:
        raise ValueError(
            f"path_readout_mode must be one of {sorted(valid_path_readout_modes)}, got {path_readout_mode!r}"
        )

    prior = MHLCMCRP(
        alpha=alpha,
        max_depth=max_depth,
        nParticles=nParticles,
        nMaxCauses=nMaxCauses,
        random_seed=random_seed,
        depth_decay=None,          # default keeps alpha tied to itself
        max_children=max_children,
        stickiness=omega,          # base kappa tied to omega
        stickiness_decay=stickiness_decay,  # default ties decay to omega too
        pe_adapt_stickiness=pe_adapt_stickiness,
        pe_sensitivity=pe_sensitivity,
    )

    def _tree_reward_prediction(path_row, particle_idx, outcome_feature_idx):
        """
        Reward prediction from valid nodes on one sampled path.

        Two task-general readouts are supported:

        1. path_readout_mode == "weighted": reliability-weighted average over
           all nodes on the path, as in the previous version.

        2. path_readout_mode in {"most_informative", "max_confidence", "winner"}:
           choose the node on the path with the largest outcome-information score:

               score = reliability * abs(P(outcome|node) - 0.5) * level_decay**depth

           and return that node's reliability-shrunk outcome prediction. This is
           still task-general: it does not know which level corresponds to shape,
           color, rule, context, etc. It simply lets HOLMES use whichever level of
           the inferred tree currently carries the strongest outcome evidence,
           rather than averaging informative and uninformative levels back toward
           0.5.
        """
        valid_nodes = path_row[path_row >= 0]
        if len(valid_nodes) == 0:
            return 0.5

        probs = []
        weights = []
        scores = []
        for depth, node_id in enumerate(valid_nodes):
            node_id = int(node_id)
            n_o = node_nFC[outcome_feature_idx, node_id, particle_idx]
            b_o = node_bFC[outcome_feature_idx, node_id, particle_idx]
            total = max(n_o + b_o, eps)
            p_raw = n_o / total

            # Evidence reliability: low-count nodes are softly shrunk toward 0.5.
            # Subtract the initial pseudo-count mass so prior-only nodes have
            # near-zero reliability.
            evidence = max(total - (aPrior + bPrior), 0.0)
            reliability = evidence / (evidence + uncertainty_k)
            p_shrunk = reliability * p_raw + (1.0 - reliability) * 0.5

            structural_weight = level_decay ** depth
            probs.append(p_shrunk)
            weights.append(structural_weight * max(reliability, eps))
            scores.append(structural_weight * reliability * abs(p_raw - 0.5))

        weights = np.asarray(weights, dtype=float)
        probs = np.asarray(probs, dtype=float)
        scores = np.asarray(scores, dtype=float)

        if path_readout_mode in ("most_informative", "max_confidence", "winner"):
            if np.isfinite(scores).all() and scores.max() > eps:
                return float(probs[int(np.argmax(scores))])
            return float(np.mean(probs))

        # Weighted path average.
        if not np.isfinite(weights).all() or weights.sum() <= eps:
            return float(np.mean(probs))
        return float(np.dot(weights / weights.sum(), probs))

    # ============================================================
    # TRIAL LOOP
    # ============================================================

    # Previous-posterior weights used only for *next-trial* prediction.
    # These weights are aligned to the current, already-resampled particle
    # arrays. They never include the current trial's outcome, so they preserve
    # strict online prediction.
    prev_wt_for_prediction = np.ones(nParticles, dtype=float) / nParticles
    prediction_prior_mix = float(np.clip(prediction_prior_mix, 0.0, 1.0))
    
    for t in range(nTrials):
        
        # Get observation and mask
        obs = f[:, t].astype(bool)
        has_fb = bool(use_fb[t])
        obs_mask_vec = feat_mask[:, t]

        # Outcome index
        oi = (outcome_idx if outcome_idx_per_trial is None 
              else int(outcome_idx_per_trial[t]))
        non_o = all_idx[all_idx != oi]

        # --------------------------------------------------------
        # PRIOR SAMPLING (BEFORE seeing data)
        # --------------------------------------------------------
        
        assigns, path_gids = prior.generate_prior(return_paths=True)
        particles[:, t] = assigns
        paths[:, t, :] = path_gids

        # Update CRP usage counts
        for m, k in enumerate(assigns):
            nC[m, k] += 1

        # --------------------------------------------------------
        # PREDICTIONS BEFORE UPDATE
        # --------------------------------------------------------
        
        # Make predictions using CURRENT parameters (before current data).
        # Original model: leaf_prob only.
        # Tree-aware model: mix leaf prediction with reward rates from all nodes
        # on the sampled path, allowing ancestors to support transfer.
        for l in range(nParticles):
            k = particles[l, t]
            
            n_o = nFC[oi, k, l]
            b_o = bFC[oi, k, l]
            leaf_prob = n_o / max(n_o + b_o, eps)

            if use_tree_prediction:
                path_prob = _tree_reward_prediction(path_gids[l], l, oi)
                mix = float(np.clip(tree_mix, 0.0, 1.0))
                rProb[l] = (1.0 - mix) * leaf_prob + mix * path_prob
            else:
                rProb[l] = leaf_prob

        # --------------------------------------------------------
        # LIKELIHOOD COMPUTATION
        # --------------------------------------------------------
        
        log_rt = np.zeros(nParticles)
        log_wt = np.zeros(nParticles)
        
        for l in range(nParticles):
            k = particles[l, t]

            # Reward log-likelihood = evaluate all *non-outcome* features
            log_rt[l] = _log_likelihood_sum(
                obs, nFC, bFC,
                cause_k=k, part_l=l,
                feat_idx=non_o,
                length_normalize=length_normalize,
                tau=tau,
                obs_mask=obs_mask_vec
            )

            # Feature log-likelihood for weighting
            feat_w = (all_idx if (has_fb and include_outcome_in_weight) 
                     else non_o)
            
            log_wt[l] = _log_likelihood_sum(
                obs, nFC, bFC,
                cause_k=k, part_l=l,
                feat_idx=feat_w,
                length_normalize=length_normalize,
                tau=tau,
                obs_mask=obs_mask_vec
            )

            # --------------------------------------------------------
            # POSTERIOR UPDATES (AFTER computing likelihood)
            # --------------------------------------------------------
            
            if has_fb:
                # Full feedback: update all observed features at the leaf.
                inc_present = obs.astype(int) * obs_mask_vec.astype(int)
                inc_absent = (1 - obs).astype(int) * obs_mask_vec.astype(int)

                nFC[:, k, l] += inc_present
                bFC[:, k, l] += inc_absent

                # Also update every valid node on the path. This is the key
                # mechanism that lets parent nodes pool evidence across leaves.
                if use_tree_prediction:
                    for node_id in path_gids[l][path_gids[l] >= 0]:
                        node_nFC[:, int(node_id), l] += inc_present
                        node_bFC[:, int(node_id), l] += inc_absent
            else:
                # No feedback: update non-outcome features only.
                nFC[non_o, k, l] += obs[non_o].astype(int)
                bFC[non_o, k, l] += (~obs[non_o]).astype(int)

                if use_tree_prediction:
                    inc_present = np.zeros(nFeatures, dtype=int)
                    inc_absent = np.zeros(nFeatures, dtype=int)
                    inc_present[non_o] = obs[non_o].astype(int)
                    inc_absent[non_o] = (~obs[non_o]).astype(int)
                    for node_id in path_gids[l][path_gids[l] >= 0]:
                        node_nFC[:, int(node_id), l] += inc_present
                        node_bFC[:, int(node_id), l] += inc_absent

        # --------------------------------------------------------
        # COMPUTE PARTICLE WEIGHTS (FROM LOG-LIKELIHOODS)
        # --------------------------------------------------------
        
        # Normalize in log space for numerical stability
        max_log_wt = np.max(log_wt)
        wt = np.exp(log_wt - max_log_wt)
        wt = wt / np.maximum(wt.sum(), eps)
        
        # Normalize prediction weights based only on non-outcome features.
        # These are valid for online prediction because they exclude the
        # current trial's outcome.
        max_log_rt = np.max(log_rt)
        rt = np.exp(log_rt - max_log_rt)
        rt = rt / np.maximum(rt.sum(), eps)

        # Optional valid online smoothing: combine current non-outcome evidence
        # with previous-trial posterior mass. The previous weights were computed
        # after observing trial t-1 and aligned to the current resampled particle
        # set, so they do not leak the current outcome.
        if prediction_prior_mix > 0.0:
            rt = ((1.0 - prediction_prior_mix) * rt +
                  prediction_prior_mix * prev_wt_for_prediction)
            rt = rt / np.maximum(rt.sum(), eps)
        
        # Safety check
        if not np.all(np.isfinite(wt)) or not np.all(wt >= 0):
            # print(f"WARNING at trial {t}: Invalid weights detected!")
            # print(f"  log_wt range: [{np.min(log_wt):.2f}, {np.max(log_wt):.2f}]")
            # print(f"  wt range: [{np.min(wt):.2e}, {np.max(wt):.2e}]")
            # print(f"  wt sum: {wt.sum():.2e}")
            # Reset to uniform if something went wrong
            wt = np.ones(nParticles) / nParticles
            rt = np.ones(nParticles) / nParticles

        # --------------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------------
        if return_rprob:
            rProb_parts[:,t] = rProb.copy()
        # Strict online prediction: use rt, not wt.
        # wt conditions on the current outcome when include_outcome_in_weight=True
        # and would therefore leak the answer into the prediction.
        rEst[t] = float(np.dot(rt, rProb))
        cEst[:, t] = weighted_hist(particles[:, t], wt, nMaxCauses)

        # Store particle-wise outcome prediction error to modulate next-trial
        # persistence. This is task-general change-point pressure: after a
        # surprising outcome, the next prior is less sticky. If outcome feedback
        # is not available, do not inject surprise.
        if pe_adapt_stickiness and has_fb and obs_mask_vec[oi]:
            current_prediction_error = np.abs(float(obs[oi]) - rProb)
        else:
            current_prediction_error = np.zeros(nParticles, dtype=float)

        # # Diagnostic output
        # if t % 10 == 0:
        #     unique_causes = len(np.unique(particles[:, t]))
        #     # Check how many times each cause has been used
        #     cause_counts = np.bincount(particles[:, :t+1].flatten())
        #     reused_causes = np.sum(cause_counts > 1)
            
        #     print(f"Trial {t:3d}: "
        #           f"causes={unique_causes}/{reused_causes} | "
        #           f"rProb μ={rProb.mean():.3f} σ={rProb.std():.3f} | "
        #           f"wt μ={wt.mean():.4f} σ={wt.std():.4f} | "
        #           f"log_wt μ={log_wt.mean():.1f} σ={log_wt.std():.1f}")

        # Hierarchy posterior
        for L in range(max_depth):
            valid = paths[:, t, L] >= 0
            if np.any(valid):
                cEst_levels[L, :, t] = weighted_hist(
                    paths[valid, t, L], wt[valid], nMaxCauses
                )

        # Feature expectations
        for i in range(nFeatures):
            ass = particles[:, t]
            n_curr = nFC[i, ass, np.arange(nParticles)]
            b_curr = bFC[i, ass, np.arange(nParticles)]
            phiEst[i, t] = float(np.dot(wt, n_curr / np.maximum(n_curr + b_curr, eps)))

        # --------------------------------------------------------
        # RESAMPLING
        # --------------------------------------------------------
        if 'weight_hist' not in locals():
            weight_hist = []
        weight_hist.append(wt.copy())

        if pe_adapt_stickiness:
            prior.set_prediction_error(current_prediction_error)

        idx = rng.choice(nParticles, nParticles, p=wt)

        # Align previous posterior weights to the post-resampling particle array
        # for use on the next trial. This is optional heuristic smoothing of the
        # prediction weights; it uses only past outcomes and therefore remains
        # valid for online prediction. Normalize after indexing because the
        # sampled ancestor weights need not sum to one.
        prev_wt_for_prediction = wt[idx].copy()
        prev_wt_for_prediction = prev_wt_for_prediction / np.maximum(
            prev_wt_for_prediction.sum(), eps
        )

        particles = particles[idx]
        nC = nC[idx]
        nFC = nFC[:, :, idx]
        bFC = bFC[:, :, idx]
        node_nFC = node_nFC[:, :, idx]
        node_bFC = node_bFC[:, :, idx]
        paths = paths[idx]  # resample paths too!
        prior.resample(idx)

    # ============================================================
    # POST-PROCESSING: REINDEX PATHS
    # ============================================================
    
    # Create dense indexing for each level
    level_gid_to_dense = []
    for L in range(max_depth):
        used = paths[:, :, L].flatten()
        used = used[used >= 0]
        unique = sorted(set(used))
        level_gid_to_dense.append({g: i for i, g in enumerate(unique)})

    # Reindex paths
    paths_dense = -np.ones_like(paths)
    for m in range(nParticles):
        for t in range(nTrials):
            for L in range(max_depth):
                g = paths[m, t, L]
                if g >= 0 and g in level_gid_to_dense[L]:
                    paths_dense[m, t, L] = level_gid_to_dense[L][g]

    # Reindex level estimates
    cEst_levels_dense = np.zeros_like(cEst_levels)
    for L in range(max_depth):
        mapping = level_gid_to_dense[L]
        K = len(mapping)
        for t in range(nTrials):
            valid = paths_dense[:, t, L] >= 0
            if np.any(valid):
                cEst_levels_dense[L, :K, t] = weighted_hist(
                    paths_dense[valid, t, L], wt[valid], K
                )
    if return_rprob:
        return particles, cEst, rEst, phiEst, cEst_levels_dense, paths_dense, weight_hist, rProb_parts
    else:
        return particles, cEst, rEst, phiEst, cEst_levels_dense, paths_dense, weight_hist