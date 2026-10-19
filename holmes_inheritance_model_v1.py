"""
HOLMES-style online hierarchical structure learner with categorical feature-family
inheritance.

This model is an extension of the HOLMES / nested-CRP particle-filter idea.  It
retains:
    * online trial-by-trial inference over variable-depth paths;
    * level-pooled CRP branch statistics;
    * depth-decayed branching concentration;
    * node-specific learned stopping;
    * finite-candidate (FC) optimal resampling with propagated particle weights.

The main extension is the observation model.  Instead of storing feature
statistics only at terminal causes, EVERY visited node stores categorical counts
for each feature family.  A child's predictive distribution shrinks toward its
parent:

    p_v(k) = [n_v(k) + lambda * p_parent(k)] / [N_v + lambda]

where lambda = rho / (1-rho).  The root uses a symmetric Dirichlet(omega)
prior.  Thus higher nodes can pool regularities shared by descendants while
lower nodes specialize only when data justify refinement.

rho is the one new inheritance parameter in the minimal version:
    rho -> 0 : weak inheritance / rapid local specialization
    rho -> 1 : strong inheritance / conservative refinement

Categorical feature families are used directly (e.g., alien identity: 8-way,
top action: 2-way, middle action: 4-way, bottom action: 8-way), avoiding the
artifact of treating one-hot categories as independent Bernoulli zeros.

Backward compatibility with HOLMES depth decay is preserved:
    depth_decay=None  ->  depth_decay=alpha.

The code intentionally leaves max_depth and max_children configurable so the
same architecture can later express under-deep and over-deep solutions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-12


class GlobalNodeRegistry:
    """Canonical global node identities shared across particles.

    Node 0 is a synthetic root.  Actual level-0 latent nodes are children of
    node 0 and therefore receive positive IDs.
    """

    def __init__(self, max_children: int = 5, max_nodes: Optional[int] = None):
        self.max_children = int(max_children)
        self.max_nodes = max_nodes
        self.registry: Dict[Tuple[int, Optional[int], int], int] = {( -1, None, 0): 0}
        self.parent: Dict[int, Optional[int]] = {0: None}
        self.level: Dict[int, int] = {0: -1}
        self.children: Dict[int, List[int]] = {0: []}
        self.next_id = 1

    def get_or_create(self, level: int, parent_id: int, branch_index: int) -> int:
        sig = (int(level), int(parent_id), int(branch_index))
        if sig in self.registry:
            return self.registry[sig]
        gid = self.next_id
        if self.max_nodes is not None and gid >= self.max_nodes:
            raise RuntimeError(
                f"max_nodes={self.max_nodes} exceeded while enumerating a candidate "
                f"node (level={level}, parent={parent_id}, branch={branch_index})."
            )
        self.next_id += 1
        self.registry[sig] = gid
        self.parent[gid] = int(parent_id)
        self.level[gid] = int(level)
        self.children.setdefault(int(parent_id), []).append(gid)
        self.children[gid] = []
        return gid


def weighted_hist(assigns: np.ndarray, weights: np.ndarray, nbins: int) -> np.ndarray:
    out = np.zeros(nbins, dtype=float)
    np.add.at(out, assigns, weights)
    return out


def enumerate_paths_nodestop(
    level_counts: List[List[List[int]]],
    s_stop: np.ndarray,
    s_cont: np.ndarray,
    global_registry: GlobalNodeRegistry,
    n_particles: int,
    max_depth: int,
    alpha: float,
    depth_decay: float,
    max_children: int,
    kappa: float,
):
    """Enumerate every stop/continue + branch candidate for every particle."""

    alpha_levels = alpha * np.exp(-depth_decay * np.arange(max_depth))

    counts_pad = np.zeros((n_particles, max_depth, max_children), dtype=float)
    K = np.zeros((n_particles, max_depth), dtype=int)
    for p in range(n_particles):
        for L in range(max_depth):
            c = level_counts[p][L]
            if len(c) > max_children:
                raise RuntimeError("level_counts exceeded max_children")
            K[p, L] = len(c)
            if c:
                counts_pad[p, L, : len(c)] = c

    frontier_particle = np.arange(n_particles)
    frontier_branch_path = np.zeros((n_particles, 0), dtype=int)
    frontier_gids = np.zeros((n_particles, 0), dtype=int)
    frontier_logp = np.zeros(n_particles, dtype=float)
    frontier_node = np.zeros(n_particles, dtype=int)  # synthetic root

    results_particle, results_branch, results_gids, results_logp = [], [], [], []

    for L in range(max_depth):
        if len(frontier_particle) == 0:
            break
        p_idx = frontier_particle
        n_front = len(p_idx)

        if L > 0:
            prior_stop = 1.0 / (1.0 + alpha_levels[L - 1])
            ss = s_stop[p_idx, frontier_node]
            sc = s_cont[p_idx, frontier_node]
            stop_p = (ss + kappa * prior_stop) / np.maximum(ss + sc + kappa, _EPS)

            # Stopping at the node already reached is a valid candidate.
            results_particle.append(p_idx.copy())
            results_branch.append(frontier_branch_path.copy())
            results_gids.append(frontier_gids.copy())
            results_logp.append(frontier_logp + np.log(np.maximum(stop_p, 1e-300)))
            continue_logp = frontier_logp + np.log(np.maximum(1.0 - stop_p, 1e-300))
        else:
            continue_logp = frontier_logp

        Kl = K[p_idx, L]
        counts = counts_pad[p_idx, L, :]
        can_create = Kl < max_children
        # When the hard child cap is reached, impossible new-child mass is
        # removed and existing branches renormalize to one.
        total = counts.sum(axis=1) + alpha_levels[L] * can_create.astype(float)

        J = np.arange(max_children)[None, :]
        is_existing = J < Kl[:, None]
        is_new = (J == Kl[:, None]) & can_create[:, None]
        valid = is_existing | is_new

        logprob = np.full((n_front, max_children), -np.inf, dtype=float)
        total_tiled = np.repeat(total, max_children).reshape(n_front, max_children)
        logprob[is_existing] = np.log(counts[is_existing] / total_tiled[is_existing])
        new_lp = np.log(alpha_levels[L] / total)
        new_lp_tiled = np.repeat(new_lp, max_children).reshape(n_front, max_children)
        logprob[is_new] = new_lp_tiled[is_new]

        fi, bj = np.where(valid)
        new_gids = np.empty(len(fi), dtype=int)
        for j in range(len(fi)):
            new_gids[j] = global_registry.get_or_create(
                L, int(frontier_node[fi[j]]), int(bj[j])
            )

        frontier_particle = p_idx[fi]
        frontier_branch_path = np.concatenate(
            [frontier_branch_path[fi], bj[:, None]], axis=1
        )
        frontier_gids = np.concatenate([frontier_gids[fi], new_gids[:, None]], axis=1)
        frontier_node = new_gids
        frontier_logp = continue_logp[fi] + logprob[fi, bj]

    # Reaching max_depth is a terminal candidate too.
    if len(frontier_particle):
        results_particle.append(frontier_particle)
        results_branch.append(frontier_branch_path)
        results_gids.append(frontier_gids)
        results_logp.append(frontier_logp)

    cand_particle = np.concatenate(results_particle)
    cand_logp = np.concatenate(results_logp)
    M = len(cand_particle)
    cand_branch = -np.ones((M, max_depth), dtype=int)
    cand_gids = -np.ones((M, max_depth), dtype=int)
    offset = 0
    for bp, gp in zip(results_branch, results_gids):
        n = len(bp)
        if bp.shape[1] > 0:
            cand_branch[offset : offset + n, : bp.shape[1]] = bp
            cand_gids[offset : offset + n, : gp.shape[1]] = gp
        offset += n

    return cand_particle, cand_branch, cand_gids, cand_logp


def fc_threshold(weights: np.ndarray, N: int):
    """Finite-candidate optimal-resampling threshold."""
    order = np.argsort(-weights)
    sorted_w = weights[order]
    M = len(weights)
    csum_tail = np.concatenate([np.cumsum(sorted_w[::-1])[::-1], [0.0]])

    n_keep, c = 0, np.inf
    for nk in range(M + 1):
        tail_sum = csum_tail[nk]
        c_try = np.inf if tail_sum <= 1e-15 else (N - nk) / tail_sum
        ok = True
        if nk > 0 and c_try * sorted_w[nk - 1] < 1 - 1e-9:
            ok = False
        if nk < M and np.isfinite(c_try) and c_try * sorted_w[nk] >= 1 + 1e-9:
            ok = False
        if ok:
            n_keep, c = nk, c_try
            break

    keep = np.zeros(M, dtype=bool)
    keep[order[:n_keep]] = True
    return c, keep, n_keep


def stratified_resample(probs: np.ndarray, n_select: int, rng: np.random.RandomState):
    """True stratified resampling: independent jitter in every stratum."""
    if n_select <= 0:
        return np.array([], dtype=int)
    cumw = np.cumsum(probs)
    positions = (np.arange(n_select) + rng.uniform(size=n_select)) / n_select
    idx = np.searchsorted(cumw, positions, side="right")
    return np.clip(idx, 0, len(probs) - 1)


def _rho_to_lambda(rho: float) -> float:
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if rho == 0.0:
        return 0.0
    return rho / max(1.0 - rho, 1e-12)


def _prepare_family_weights(family_weights, n_families: int, n_trials: int) -> np.ndarray:
    if family_weights is None:
        return np.ones((n_families, n_trials), dtype=float)
    w = np.asarray(family_weights, dtype=float)
    if w.ndim == 0:
        return np.full((n_families, n_trials), float(w))
    if w.ndim == 1:
        if len(w) != n_families:
            raise ValueError("1-D family_weights must have one value per family")
        return np.repeat(w[:, None], n_trials, axis=1)
    if w.shape != (n_families, n_trials):
        raise ValueError(
            f"family_weights must be scalar, length {n_families}, or shape "
            f"{(n_families, n_trials)}; got {w.shape}"
        )
    return w.copy()


def _node_predictive(
    particle: int,
    node: int,
    family: int,
    family_counts: Sequence[np.ndarray],
    family_sizes: Sequence[int],
    registry: GlobalNodeRegistry,
    omega: float,
    inheritance_lambda: float,
    cache: dict,
) -> np.ndarray:
    """Recursive hierarchical Dirichlet predictive for one node/family."""
    key = (particle, node, family)
    if key in cache:
        return cache[key]

    K = family_sizes[family]
    counts = family_counts[family][particle, node, :]
    if node == 0:
        vec = (counts + omega) / np.maximum(counts.sum() + omega * K, _EPS)
    else:
        parent = registry.parent[node]
        parent_vec = _node_predictive(
            particle,
            int(parent),
            family,
            family_counts,
            family_sizes,
            registry,
            omega,
            inheritance_lambda,
            cache,
        )
        if inheritance_lambda <= _EPS:
            # With no inherited pseudocount and no local data, fall back to the
            # parent rather than producing an undefined 0/0 predictive.
            if counts.sum() <= _EPS:
                vec = parent_vec
            else:
                vec = counts / counts.sum()
        else:
            vec = (counts + inheritance_lambda * parent_vec) / (
                counts.sum() + inheritance_lambda
            )
    cache[key] = vec
    return vec


def _validate_categorical_data(data: np.ndarray, family_sizes: Sequence[int]):
    if data.ndim != 2:
        raise ValueError("data must have shape (n_families, n_trials)")
    if data.shape[0] != len(family_sizes):
        raise ValueError("family_sizes length must equal data.shape[0]")
    for q, K in enumerate(family_sizes):
        vals = data[q]
        if np.any(vals < 0) or np.any(vals >= K):
            raise ValueError(f"family {q} contains values outside 0..{K-1}")


def full_hier_inference_categorical(
    data: np.ndarray,
    family_sizes: Sequence[int],
    feedback_mask: np.ndarray,
    prediction_mask: Optional[np.ndarray] = None,
    *,
    alpha: float = 1.0,
    omega: float = 0.5,
    rho: float = 0.75,
    depth_decay: Optional[float] = None,
    kappa: float = 1.0,
    family_weights=None,
    n_particles: int = 100,
    max_depth: int = 5,
    max_children: int = 5,
    random_seed: Optional[int] = None,
    max_nodes: Optional[int] = None,
):
    """Run online hierarchical inference for categorical feature families.

    Parameters
    ----------
    data : int array, shape (Q, T)
        Integer category code for each feature family on each trial.
    family_sizes : sequence of int
        Number of categories in each family.
    feedback_mask : bool array, shape (Q, T)
        Features available to posterior updating after feedback.
    prediction_mask : bool array, shape (Q, T), optional
        Features available before feedback.  If None, prediction uses the same
        information as feedback.  For the alien task this is identity-only.
    alpha, omega : float
        HOLMES-style branching concentration and Dirichlet pseudocount.
    rho : float in [0,1)
        Parent-to-child inheritance strength, transformed to an equivalent
        parent pseudocount lambda=rho/(1-rho).
    depth_decay : float or None
        If None, defaults to alpha (HOLMES convention).
    kappa : float
        Strength of the depth prior in learned node-specific stop/continue
        beliefs.
    family_weights : scalar, (Q,), or (Q,T) array
        Fixed salience / repeated-feature-equivalent weight in candidate
        likelihoods.  Counts themselves are NEVER multiplied by these weights.
    """

    data = np.asarray(data, dtype=int)
    family_sizes = [int(k) for k in family_sizes]
    _validate_categorical_data(data, family_sizes)
    Q, T = data.shape

    feedback_mask = np.asarray(feedback_mask, dtype=bool)
    if feedback_mask.shape != (Q, T):
        raise ValueError(f"feedback_mask must have shape {(Q,T)}")
    if prediction_mask is None:
        prediction_mask = feedback_mask.copy()
    else:
        prediction_mask = np.asarray(prediction_mask, dtype=bool)
        if prediction_mask.shape != (Q, T):
            raise ValueError(f"prediction_mask must have shape {(Q,T)}")
        if np.any(prediction_mask & ~feedback_mask):
            raise ValueError("prediction_mask cannot reveal a family absent from feedback_mask")

    if alpha <= 0 or omega <= 0 or kappa <= 0:
        raise ValueError("alpha, omega, and kappa must be positive")
    if n_particles <= 0 or max_depth <= 0 or max_children <= 0:
        raise ValueError("n_particles, max_depth, and max_children must be positive")

    depth_decay_val = alpha if depth_decay is None else float(depth_decay)
    inheritance_lambda = _rho_to_lambda(float(rho))
    fam_weight = _prepare_family_weights(family_weights, Q, T)
    rng = np.random.RandomState(random_seed)

    if max_nodes is None:
        max_nodes = max(512, min(T, 120) * max_depth * max_children + 32)

    registry = GlobalNodeRegistry(max_children=max_children, max_nodes=max_nodes)
    level_counts = [[[] for _ in range(max_depth)] for _ in range(n_particles)]
    s_stop = np.zeros((n_particles, max_nodes), dtype=float)
    s_cont = np.zeros((n_particles, max_nodes), dtype=float)

    # One categorical count cube per family: (particle, node, category).
    family_counts = [
        np.zeros((n_particles, max_nodes, K), dtype=float) for K in family_sizes
    ]

    particle_weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    paths = -np.ones((n_particles, T, max_depth), dtype=int)
    path_lengths = np.zeros((n_particles, T), dtype=int)
    pred_probs = [np.zeros((T, K), dtype=float) for K in family_sizes]
    c_est_levels = np.zeros((max_depth, max_nodes, T), dtype=float)
    ess = np.zeros(T, dtype=float)

    for t in range(T):
        cand_particle, cand_branch, cand_gids, cand_logp = enumerate_paths_nodestop(
            level_counts,
            s_stop,
            s_cont,
            registry,
            n_particles,
            max_depth,
            alpha,
            depth_decay_val,
            max_children,
            kappa,
        )

        # Deterministic ordering aids reproducibility.
        sort_keys = [cand_branch[:, L] for L in range(max_depth - 1, -1, -1)]
        sort_keys.append(cand_particle)
        order = np.lexsort(sort_keys)
        cand_particle = cand_particle[order]
        cand_branch = cand_branch[order]
        cand_gids = cand_gids[order]
        cand_logp = cand_logp[order]

        M = len(cand_particle)
        plen = (cand_branch >= 0).sum(axis=1)
        leaf = np.array([cand_gids[i, plen[i] - 1] for i in range(M)], dtype=int)
        log_parent_w = np.log(np.maximum(particle_weights[cand_particle], 1e-300))

        cache = {}
        candidate_preds = [np.zeros((M, K), dtype=float) for K in family_sizes]
        pred_loglik = np.zeros(M, dtype=float)
        full_loglik = np.zeros(M, dtype=float)

        for i in range(M):
            p = int(cand_particle[i])
            node = int(leaf[i])
            for q, K in enumerate(family_sizes):
                vec = _node_predictive(
                    p,
                    node,
                    q,
                    family_counts,
                    family_sizes,
                    registry,
                    omega,
                    inheritance_lambda,
                    cache,
                )
                candidate_preds[q][i, :] = vec
                val = int(data[q, t])
                wt = float(fam_weight[q, t])
                if prediction_mask[q, t]:
                    pred_loglik[i] += wt * np.log(max(vec[val], 1e-300))
                if feedback_mask[q, t]:
                    full_loglik[i] += wt * np.log(max(vec[val], 1e-300))

        # Genuine pre-feedback prediction distribution.
        pred_logw = log_parent_w + cand_logp + pred_loglik
        pred_logw -= pred_logw.max()
        w_pred = np.exp(pred_logw)
        w_pred /= np.maximum(w_pred.sum(), _EPS)
        for q in range(Q):
            pred_probs[q][t, :] = w_pred @ candidate_preds[q]

        # Posterior used for learning / resampling after feedback.
        logw = log_parent_w + cand_logp + full_loglik
        logw -= logw.max()
        w = np.exp(logw)
        w /= np.maximum(w.sum(), _EPS)
        ess[t] = 1.0 / np.maximum(np.sum(w * w), _EPS)

        c_thresh, keep_mask, n_keep = fc_threshold(w, n_particles)
        kept_idx = np.where(keep_mask)[0]
        tail_idx = np.where(~keep_mask)[0]
        n_tail = n_particles - n_keep
        if n_tail > 0 and len(tail_idx) > 0:
            tail_w = w[tail_idx]
            tail_probs = tail_w / np.maximum(tail_w.sum(), _EPS)
            sel = stratified_resample(tail_probs, n_tail, rng)
            tail_selected = tail_idx[sel]
        else:
            tail_selected = np.array([], dtype=int)

        final_idx = np.concatenate([kept_idx, tail_selected])
        final_weights = np.concatenate(
            [
                w[kept_idx],
                np.full(
                    len(tail_selected),
                    1.0 / c_thresh if np.isfinite(c_thresh) else 0.0,
                ),
            ]
        )
        final_weights /= np.maximum(final_weights.sum(), _EPS)

        new_level_counts = []
        new_s_stop = np.zeros_like(s_stop)
        new_s_cont = np.zeros_like(s_cont)
        new_family_counts = [np.zeros_like(x) for x in family_counts]
        new_paths = -np.ones((n_particles, max_depth), dtype=int)
        new_plen = np.zeros(n_particles, dtype=int)

        for new_p, ci in enumerate(final_idx):
            old_p = int(cand_particle[ci])
            pl = int(plen[ci])
            bp = cand_branch[ci, :pl]
            gp = cand_gids[ci, :pl]

            lc = [list(x) for x in level_counts[old_p]]
            for L, branch_idx in enumerate(bp):
                b = int(branch_idx)
                if b == len(lc[L]):
                    lc[L].append(1)
                else:
                    lc[L][b] += 1
            new_level_counts.append(lc)

            new_s_stop[new_p, :] = s_stop[old_p, :]
            new_s_cont[new_p, :] = s_cont[old_p, :]
            for q in range(Q):
                new_family_counts[q][new_p, :, :] = family_counts[q][old_p, :, :]

            # Every node passed through is evidence for continuation.  If the
            # candidate voluntarily stopped before max_depth, that terminal node
            # gets a stop increment.
            for L in range(1, pl):
                new_s_cont[new_p, int(gp[L - 1])] += 1.0
            if 0 < pl < max_depth:
                new_s_stop[new_p, int(gp[pl - 1])] += 1.0

            # Hierarchical inheritance update: root and EVERY visited node see
            # the observation, so ancestors pool descendant regularities.
            update_nodes = np.concatenate(([0], gp.astype(int)))
            for q in range(Q):
                if feedback_mask[q, t]:
                    val = int(data[q, t])
                    new_family_counts[q][new_p, update_nodes, val] += 1.0

            new_paths[new_p, :pl] = gp
            new_plen[new_p] = pl

        level_counts = new_level_counts
        s_stop, s_cont = new_s_stop, new_s_cont
        family_counts = new_family_counts
        particle_weights = final_weights.copy()
        paths[:, t, :] = new_paths
        path_lengths[:, t] = new_plen

        for L in range(max_depth):
            valid = new_paths[:, L] >= 0
            if np.any(valid):
                c_est_levels[L, :, t] = weighted_hist(
                    new_paths[valid, L], final_weights[valid], max_nodes
                )

    return {
        "pred_probs": pred_probs,
        "paths": paths,
        "path_lengths": path_lengths,
        "particle_weights": particle_weights,
        "level_counts": level_counts,
        "s_stop": s_stop,
        "s_cont": s_cont,
        "family_counts": family_counts,
        "c_est_levels": c_est_levels,
        "ess": ess,
        "registry": registry,
        "family_sizes": family_sizes,
        "params": {
            "alpha": float(alpha),
            "omega": float(omega),
            "rho": float(rho),
            "inheritance_lambda": float(inheritance_lambda),
            "depth_decay": float(depth_decay_val),
            "kappa": float(kappa),
            "n_particles": int(n_particles),
            "max_depth": int(max_depth),
            "max_children": int(max_children),
            "max_nodes": int(max_nodes),
        },
    }


def probe_categorical(
    fit: dict,
    values: Sequence[int],
    observed_mask: Sequence[bool],
    family_weights=None,
):
    """Posterior prediction/path distribution without modifying the fitted state.

    This is useful for post-training structural probes.  ``values`` supplies a
    category code for every family, while ``observed_mask`` says which of those
    values are available for routing/conditioning on the probe.
    """

    values = np.asarray(values, dtype=int)
    mask = np.asarray(observed_mask, dtype=bool)
    family_sizes = fit["family_sizes"]
    Q = len(family_sizes)
    if values.shape != (Q,) or mask.shape != (Q,):
        raise ValueError(f"values and observed_mask must both have shape {(Q,)}")
    for q, K in enumerate(family_sizes):
        if not (0 <= values[q] < K):
            raise ValueError(f"probe family {q} value outside 0..{K-1}")

    params = fit["params"]
    registry = fit["registry"]
    n_particles = params["n_particles"]
    max_depth = params["max_depth"]

    cand_particle, cand_branch, cand_gids, cand_logp = enumerate_paths_nodestop(
        fit["level_counts"],
        fit["s_stop"],
        fit["s_cont"],
        registry,
        n_particles,
        max_depth,
        params["alpha"],
        params["depth_decay"],
        params["max_children"],
        params["kappa"],
    )

    sort_keys = [cand_branch[:, L] for L in range(max_depth - 1, -1, -1)]
    sort_keys.append(cand_particle)
    order = np.lexsort(sort_keys)
    cand_particle = cand_particle[order]
    cand_branch = cand_branch[order]
    cand_gids = cand_gids[order]
    cand_logp = cand_logp[order]

    plen = (cand_branch >= 0).sum(axis=1)
    leaf = np.array([cand_gids[i, plen[i] - 1] for i in range(len(plen))], dtype=int)
    parent_logw = np.log(np.maximum(fit["particle_weights"][cand_particle], 1e-300))

    w_family = _prepare_family_weights(family_weights, Q, 1)[:, 0]
    cache = {}
    candidate_preds = [np.zeros((len(plen), K), dtype=float) for K in family_sizes]
    loglik = np.zeros(len(plen), dtype=float)
    for i in range(len(plen)):
        p = int(cand_particle[i])
        node = int(leaf[i])
        for q, K in enumerate(family_sizes):
            vec = _node_predictive(
                p,
                node,
                q,
                fit["family_counts"],
                family_sizes,
                registry,
                params["omega"],
                params["inheritance_lambda"],
                cache,
            )
            candidate_preds[q][i, :] = vec
            if mask[q]:
                loglik[i] += w_family[q] * np.log(max(vec[int(values[q])], 1e-300))

    logw = parent_logw + cand_logp + loglik
    logw -= logw.max()
    w = np.exp(logw)
    w /= np.maximum(w.sum(), _EPS)

    family_probs = [w @ candidate_preds[q] for q in range(Q)]
    path_mass = defaultdict(float)
    for i in range(len(plen)):
        path_mass[tuple(int(x) for x in cand_gids[i, : plen[i]])] += float(w[i])
    path_mass = dict(sorted(path_mass.items(), key=lambda kv: -kv[1]))

    level_marginals = []
    reach_mass = []
    conditional_level_marginals = []
    for L in range(max_depth):
        valid = plen > L
        reach = float(w[valid].sum())
        reach_mass.append(reach)
        marg = defaultdict(float)
        for i in np.where(valid)[0]:
            marg[int(cand_gids[i, L])] += float(w[i])
        level_marginals.append(dict(marg))
        if reach > _EPS:
            conditional_level_marginals.append({k: v / reach for k, v in marg.items()})
        else:
            conditional_level_marginals.append({})

    dominant_signature = []
    conditional_confidence = []
    for L, marg in enumerate(conditional_level_marginals):
        if not marg:
            dominant_signature.append(None)
            conditional_confidence.append(0.0)
        else:
            node, mass = max(marg.items(), key=lambda kv: kv[1])
            dominant_signature.append(int(node))
            conditional_confidence.append(float(mass))

    return {
        "family_probs": family_probs,
        "path_mass": path_mass,
        "reach_mass": np.asarray(reach_mass),
        "level_marginals": level_marginals,
        "conditional_level_marginals": conditional_level_marginals,
        "dominant_signature": tuple(dominant_signature),
        "conditional_confidence": np.asarray(conditional_confidence),
    }


# Backward-friendly short alias for notebooks.
run_model = full_hier_inference_categorical
