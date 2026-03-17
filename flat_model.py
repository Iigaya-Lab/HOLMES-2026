# -------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------
# CRP with seeded RNG
# -------------------------------------------------------------------

class CRP:
    def __init__(self, alpha, rng=None):
        self.alpha = alpha
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_prior(self, nKTminus1):
        """
        Standard CRP prior for new assignments.
        nKTminus1: (M x K) table of counts

        Returns:
            assignmentsT: (M,)
            nKT: updated counts
        """
        M, Kmax = nKTminus1.shape
        assignmentsT = np.zeros(M, dtype=int)
        nKT = nKTminus1.copy()

        for m in range(M):
            # Count active causes
            K = np.sum(nKTminus1[m] > 0)

            # Existing causes + new cause
            probs = np.append(nKTminus1[m, :K], self.alpha)
            probs = np.append(probs, np.zeros(Kmax - K - 1))
            probs = probs / probs.sum()

            # FIX: Use local RNG instead of global np.random
            k_new = self.rng.choice(len(probs), p=probs)
            assignmentsT[m] = k_new
            nKT[m, k_new] += 1

        return assignmentsT, nKT



# ------------------ Helper functions ------------------

def weighted_hist(labels, weights, K):
    out = np.zeros(K)
    for lab, w in zip(labels, weights):
        if 0 <= lab < K:
            out[int(lab)] += w
    return out


def _bernoulli_predictive(obs_bool_vec, n_vec, b_vec, eps=1e-12):
    denom = np.maximum(n_vec + b_vec, eps)
    p1 = n_vec / denom
    p0 = b_vec / denom
    O = obs_bool_vec.astype(float)
    return O * p1 + (1 - O) * p0


def _likelihood_product(obs_bool, nFC, bFC, k, particle_idx, feat_idx,
                        length_normalize=False, tau=1.0, eps=1e-12):

    if len(feat_idx) == 0:
        return 1.0

    preds = _bernoulli_predictive(
        obs_bool[feat_idx],
        nFC[feat_idx, k, particle_idx],
        bFC[feat_idx, k, particle_idx],
        eps=eps
    )

    preds = np.maximum(preds, eps)

    if length_normalize:
        return float(np.exp(tau * np.mean(np.log(preds))))
    else:
        return float(np.prod(preds))
    
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
#               ONE-LAYER INFERENCE LOOP
# ============================================================

def one_layer_inference_loop(
    nTrials,
    nParticles,
    nFeatures,
    regime,
    alpha,
    f,
    outcome_idx_per_trial,
    feedback_mask=None,
    include_outcome_in_weight=True,
    length_normalize=False,
    tau=1.0,
    random_seed=0,
    return_rprob=False
):
    """
    Simplest possible one-layer CRP LC model
    based on Gershman and Niv
    """

    rng = np.random.default_rng(random_seed)

    # ----------------------------------------------
    # Ensure feedback mask is proper
    # ----------------------------------------------
    if feedback_mask is None:
        feedback_mask = np.ones(nTrials, dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask, dtype=bool)
        assert feedback_mask.ndim == 1
        assert len(feedback_mask) == nTrials

    # ----------------------------------------------
    # Allocate storage
    # ----------------------------------------------
    nMaxCauses = nTrials

    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC  = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime

    wt = np.ones(nParticles) / nParticles
    rt = np.ones(nParticles) / nParticles
    rProb = np.zeros(nParticles)
    rProb_parts= np.zeros((nParticles, nTrials))

    cEst   = np.zeros((nMaxCauses, nTrials))
    rEst   = np.zeros(nTrials)
    phiEst = np.zeros((nFeatures, nTrials))

    # Pass RNG to CRP
    crp = CRP(alpha, rng=rng)
    all_idx = np.arange(nFeatures)

    # ----------------------------------------------
    # Trial loop
    # ----------------------------------------------
    for t in range(nTrials):

        obs = f[:, t].astype(bool)
        outcome_dim = int(outcome_idx_per_trial[t])
        non_outcome_dims = all_idx[all_idx != outcome_dim]
        has_fb = feedback_mask[t]

        # ----- sample the CRP prior-----
        if t == 0:
            particles[:, t], nC = crp.generate_prior(nC)
        else:
            particles[:, t], nC = crp.generate_prior(nC)

        # ----- Likelihoods -----
        for p in range(nParticles):
            k = particles[p, t]

            # Prediction on non-outcome dims
            rt[p] = _likelihood_product(
                obs, nFC, bFC, k, p, non_outcome_dims,
                length_normalize, tau
            )

            # Predicted outcome
            n_o = nFC[outcome_dim, k, p]
            b_o = bFC[outcome_dim, k, p]
            rProb[p] = n_o / max(n_o + b_o, 1e-12)

            # Weight update
            if has_fb and include_outcome_in_weight:
                feat_idx = all_idx
            else:
                feat_idx = non_outcome_dims

            wt[p] = _likelihood_product(
                obs, nFC, bFC, k, p, feat_idx,
                length_normalize, tau
            )

            # ----- Update counts -----
            if has_fb:
                nFC[:, k, p] += obs
                bFC[:, k, p] += ~obs
            else:
                nFC[non_outcome_dims, k, p] += obs[non_outcome_dims]
                bFC[non_outcome_dims, k, p] += ~obs[non_outcome_dims]

        # Normalize weights
        wt /= max(wt.sum(), 1e-12)
        rt /= max(rt.sum(), 1e-12)

        # Predicted outcome
        if return_rprob:
            rProb_parts[:,t] = rProb.copy()
        rEst[t] = float(np.dot(rt, rProb))

        # Posterior on latent causes
        cEst[:, t] = weighted_hist(particles[:, t], wt, nMaxCauses)

        # Feature posteriors
        for i in range(nFeatures):
            assigned = particles[:, t]
            n_curr = nFC[i, assigned, np.arange(nParticles)]
            b_curr = bFC[i, assigned, np.arange(nParticles)]
            phiEst[i, t] = float(np.dot(wt, n_curr / np.maximum(n_curr + b_curr, 1e-12)))

        # Resample
        idx = rng.choice(nParticles, nParticles, p=wt)
        particles = particles[idx]
        nC = nC[idx]
        nFC = nFC[:, :, idx]
        bFC = bFC[:, :, idx]
    if return_rprob:
        return particles, cEst, rEst, phiEst, rProb_parts
    else:   
        return particles, cEst, rEst, phiEst