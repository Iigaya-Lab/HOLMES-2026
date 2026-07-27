#!/usr/bin/env python3
"""
HOLMES ablation + flat-model enhancement suite (self-contained).

Ablation grid (2x2 for HOLMES) + flat enhancement
-------------------------------------------------
    holmes_full          structure + stopping + stickiness   (== original HOLMES)
    holmes_no_stick      structure + stopping
    holmes_no_stop       structure +          + stickiness
    holmes_struct_only   structure                            ("just the structure"-- not really relevant)
    flat                 baseline CRP
    flat_sticky          baseline CRP + stickiness            (enhancement)

"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# SHARED LIKELIHOOD HELPERS
# all copied from the original model files 
# ============================================================

def weighted_hist(labels, weights, K):
    """Weighted histogram of integer labels into K bins."""
    out = np.zeros(K, dtype=float)
    for lab, w in zip(labels, weights):
        lab = int(lab)
        if 0 <= lab < K:
            out[lab] += w
    return out


def _bernoulli_predictive(obs_bool_vec, n_vec, b_vec, eps=1e-12):
    """Bernoulli predictive prob per feature: present -> n/(n+b), absent -> b/(n+b)."""
    denom = np.maximum(n_vec + b_vec, eps)
    p1 = n_vec / denom
    p0 = b_vec / denom
    O = obs_bool_vec.astype(np.float64)
    return O * p1 + (1.0 - O) * p0


def _likelihood_product(obs_bool, nFC, bFC, k, particle_idx, feat_idx,
                        length_normalize=False, tau=1.0, eps=1e-12):
    """Product-space likelihood used by the FLAT model (matches flat_model.py)."""
    if len(feat_idx) == 0:
        return 1.0
    preds = _bernoulli_predictive(
        obs_bool[feat_idx],
        nFC[feat_idx, k, particle_idx],
        bFC[feat_idx, k, particle_idx],
        eps=eps,
    )
    preds = np.maximum(preds, eps)
    if length_normalize:
        return float(np.exp(tau * np.mean(np.log(preds))))
    return float(np.prod(preds))


def _log_likelihood_sum(obs_bool, nFC, bFC, cause_k, part_l, feat_idx,
                        length_normalize=False, tau=1.0, obs_mask=None, eps=1e-12):
    """Log-space likelihood used by HOLMES (matches hier_model.py)."""
    if obs_mask is None:
        obs_mask = np.ones_like(obs_bool, dtype=bool)
    effective_idx = [i for i in feat_idx if obs_mask[i]]
    if len(effective_idx) == 0:
        return 0.0
    p_f = _bernoulli_predictive(
        obs_bool[effective_idx],
        nFC[effective_idx, cause_k, part_l],
        bFC[effective_idx, cause_k, part_l],
        eps=eps,
    )
    log_p_f = np.log(np.maximum(p_f, eps))
    if length_normalize:
        return float(tau * np.mean(log_p_f))
    return float(np.sum(log_p_f))


# ============================================================
# FLAT MODEL (one-layer CRP) with OPTIONAL stickiness
# ============================================================

class StickyCRP:
    """
    Flat CRP prior with an optional temporal stickiness bonus.

    stickiness == 0.0  -> standard CRP (identical draws to the original flat_model.CRP).
    stickiness  > 0.0  -> the cause chosen on the previous trial gets its CRP
                          probability multiplied by (1 + stickiness), encouraging
                          the particle to reuse the same latent cause across trials.
    """

    def __init__(self, alpha, nParticles, stickiness=0.0, rng=None):
        self.alpha = float(alpha)
        self.stickiness = float(stickiness)
        self.rng = rng if rng is not None else np.random.default_rng()
        # previous-trial assignment per particle (-1 = none yet)
        self.prev = -np.ones(int(nParticles), dtype=int)

    def generate_prior(self, nKTminus1):
        M, Kmax = nKTminus1.shape
        assignmentsT = np.zeros(M, dtype=int)
        nKT = nKTminus1.copy()
        for m in range(M):
            K = int(np.sum(nKTminus1[m] > 0))
            probs = np.append(nKTminus1[m, :K], self.alpha)
            probs = np.append(probs, np.zeros(Kmax - K - 1))

            # Temporal stickiness: bonus on the previously chosen (existing) cause.
            if self.stickiness > 0.0:
                pk = self.prev[m]
                if 0 <= pk < K:
                    probs[pk] *= (1.0 + self.stickiness)

            probs = probs / probs.sum()
            k_new = self.rng.choice(len(probs), p=probs)
            assignmentsT[m] = k_new
            nKT[m, k_new] += 1
            self.prev[m] = k_new
        return assignmentsT, nKT

    def resample(self, idx):
        """Reorder previous-assignment memory to match resampled particles."""
        self.prev = self.prev[idx]


def one_layer_inference_loop(
    nTrials, nParticles, nFeatures, regime, alpha, f,
    outcome_idx_per_trial,
    feedback_mask=None,
    include_outcome_in_weight=True,
    length_normalize=False,
    tau=1.0,
    random_seed=0,
    stickiness=0.0,          # <-- ENHANCEMENT knob (0.0 => original flat model)
    return_rprob=False,
):
    """
    One-layer CRP latent-cause model (Gershman & Niv). With stickiness=0 this is
    byte-for-byte the original flat model; stickiness>0 adds temporal cause reuse.
    """
    rng = np.random.default_rng(random_seed)

    if feedback_mask is None:
        feedback_mask = np.ones(nTrials, dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask, dtype=bool)
        assert feedback_mask.ndim == 1 and len(feedback_mask) == nTrials

    nMaxCauses = nTrials
    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime

    wt = np.ones(nParticles) / nParticles
    rt = np.ones(nParticles) / nParticles
    rProb = np.zeros(nParticles)
    rProb_parts = np.zeros((nParticles, nTrials))

    cEst = np.zeros((nMaxCauses, nTrials))
    rEst = np.zeros(nTrials)
    phiEst = np.zeros((nFeatures, nTrials))

    crp = StickyCRP(alpha, nParticles, stickiness=stickiness, rng=rng)
    all_idx = np.arange(nFeatures)

    for t in range(nTrials):
        obs = f[:, t].astype(bool)
        outcome_dim = int(outcome_idx_per_trial[t])
        non_outcome_dims = all_idx[all_idx != outcome_dim]
        has_fb = feedback_mask[t]

        particles[:, t], nC = crp.generate_prior(nC)

        for p in range(nParticles):
            k = particles[p, t]

            rt[p] = _likelihood_product(
                obs, nFC, bFC, k, p, non_outcome_dims, length_normalize, tau
            )

            n_o = nFC[outcome_dim, k, p]
            b_o = bFC[outcome_dim, k, p]
            rProb[p] = n_o / max(n_o + b_o, 1e-12)

            feat_idx = all_idx if (has_fb and include_outcome_in_weight) else non_outcome_dims
            wt[p] = _likelihood_product(
                obs, nFC, bFC, k, p, feat_idx, length_normalize, tau
            )

            if has_fb:
                nFC[:, k, p] += obs
                bFC[:, k, p] += ~obs
            else:
                nFC[non_outcome_dims, k, p] += obs[non_outcome_dims]
                bFC[non_outcome_dims, k, p] += ~obs[non_outcome_dims]

        wt /= max(wt.sum(), 1e-12)
        rt /= max(rt.sum(), 1e-12)

        if return_rprob:
            rProb_parts[:, t] = rProb.copy()
        rEst[t] = float(np.dot(rt, rProb))
        cEst[:, t] = weighted_hist(particles[:, t], wt, nMaxCauses)

        for i in range(nFeatures):
            assigned = particles[:, t]
            n_curr = nFC[i, assigned, np.arange(nParticles)]
            b_curr = bFC[i, assigned, np.arange(nParticles)]
            phiEst[i, t] = float(np.dot(wt, n_curr / np.maximum(n_curr + b_curr, 1e-12)))

        idx = rng.choice(nParticles, nParticles, p=wt)
        particles = particles[idx]
        nC = nC[idx]
        nFC = nFC[:, :, idx]
        bFC = bFC[:, :, idx]
        crp.resample(idx)          # keep stickiness memory aligned with particles

    if return_rprob:
        return particles, cEst, rEst, phiEst, rProb_parts
    return particles, cEst, rEst, phiEst


# ============================================================
# HOLMES: GLOBAL NODE REGISTRY + PER-PARTICLE TREE
# ============================================================

class GlobalNodeRegistry:
    """Canonical shared tree: identical (level,parent,branch) -> same global id."""

    def __init__(self, max_children=5):
        self.max_children = max_children
        self.registry = {}
        self.children = {}
        self.next_id = 1
        self.registry[(0, None, 0)] = 0
        self.children[None] = [0]
        self.children[0] = []

    def get_or_create(self, level, parent_id, branch_index):
        sig = (level, parent_id, branch_index)
        if sig in self.registry:
            return self.registry[sig]
        gid = self.next_id
        self.registry[sig] = gid
        self.next_id += 1
        self.children.setdefault(parent_id, []).append(gid)
        self.children[gid] = []
        return gid

    def get_children(self, parent_id):
        return self.children.get(parent_id, [])


class NCRPTreeParticle:
    """Per-particle CRP branch counts at each level."""

    def __init__(self, alpha, max_depth, max_children):
        self.alpha = float(alpha)
        self.max_depth = int(max_depth)
        self.max_children = max_children
        self.level_counts = [[] for _ in range(max_depth)]


# ============================================================
# HOLMES: HIERARCHICAL nCRP PRIOR (with lesion switches)
# ============================================================

class MHLCMCRP:
    """
    Multi-particle nested CRP prior.

    Lesion switches
    ---------------
    use_stopping : bool
        True  -> stochastic stopping (variable leaf depth)   [original]
        False -> always descend to max_depth (fixed, maximally specific leaves)
    stickiness : float
        0.0   -> no temporal path reuse
        >0.0  -> previous path's branch gets a (1+stickiness) multiplicative bonus
    """

    def __init__(self, alpha, max_depth, nParticles, nMaxCauses,
                 random_seed=None, depth_decay=1.0, max_children=5,
                 stickiness=1.0, use_stopping=True):
        self.alpha = float(alpha)
        self.max_depth = int(max_depth)
        self.nParticles = nParticles
        self.nMaxCauses = nMaxCauses
        self.depth_decay = float(depth_decay)
        self.max_children = max_children
        self.stickiness = float(stickiness)
        self.use_stopping = bool(use_stopping)

        self.rng = np.random.RandomState(random_seed)
        self.trees = [NCRPTreeParticle(alpha, max_depth, max_children)
                      for _ in range(nParticles)]
        self.leaf_map = dict()
        self.leaf_next = 0
        self.global_registry = GlobalNodeRegistry(max_children=max_children)
        self.level_gid_sets = [set() for _ in range(max_depth)]
        self.prev_path = {m: None for m in range(nParticles)}

    def resample(self, idx):
        self.trees = [self.trees[i] for i in idx]
        self.prev_path = {m: self.prev_path[i] for m, i in enumerate(idx)}

    def generate_prior(self, return_paths=False):
        assignments = np.zeros(self.nParticles, dtype=int)
        paths_global = -np.ones((self.nParticles, self.max_depth), dtype=int)
        alpha_L = self.alpha
        self.level_gid_sets = [set() for _ in range(self.max_depth)]

        for m in range(self.nParticles):
            particle = self.trees[m]
            parent_gid = 0
            path_gids = []
            prev_path = self.prev_path.get(m, None)

            for L in range(self.max_depth):
                # ---- STOPPING (lesionable) ----
                # depth 0 always continues; beyond that, stop w.p. 1/(1+alpha_L).
                if self.use_stopping and L > 0:
                    stop_prob = 1.0 / (1.0 + alpha_L)
                    if self.rng.rand() < stop_prob:
                        break

                # depth-decayed concentration
                alpha_L = self.alpha * np.exp(-self.depth_decay * L)
                counts = particle.level_counts[L]
                K = len(counts)

                if K == 0:
                    probs = np.array([1.0])
                    choice = 0
                    particle.level_counts[L].append(1)
                else:
                    if K < self.max_children:
                        probs = np.zeros(K + 1)
                        probs[:K] = np.array(counts, dtype=float)
                        probs[K] = alpha_L
                    else:
                        probs = np.array(counts, dtype=float)

                    # ---- STICKINESS (lesionable via stickiness=0) ----
                    if (self.stickiness > 0.0 and prev_path is not None
                            and L < len(prev_path) and prev_path[L] >= 0):
                        prev_gid = prev_path[L]
                        for existing in range(K):
                            candidate_gid = self.global_registry.get_or_create(
                                L, parent_gid, existing
                            )
                            if candidate_gid == prev_gid:
                                probs[existing] *= (1.0 + self.stickiness)
                                break

                    probs_norm = probs / np.sum(probs)
                    choice = self.rng.choice(len(probs_norm), p=probs_norm)

                    if choice == K and K < self.max_children:
                        particle.level_counts[L].append(1)
                    else:
                        particle.level_counts[L][choice] += 1

                gid = self.global_registry.get_or_create(L, parent_gid, choice)
                parent_gid = gid
                path_gids.append(gid)
                self.level_gid_sets[L].add(gid)

            full_path = -np.ones(self.max_depth, dtype=int)
            full_path[:len(path_gids)] = path_gids
            paths_global[m] = full_path

            leaf_gid = 0 if len(path_gids) == 0 else path_gids[-1]
            if leaf_gid not in self.leaf_map:
                leaf_id = self.leaf_next
                if leaf_id >= self.nMaxCauses:
                    raise RuntimeError(
                        f"Exceeded nMaxCauses={self.nMaxCauses} at global leaf {leaf_gid}. "
                        f"Increase n_max_causes or reduce max_depth/max_children."
                    )
                self.leaf_map[leaf_gid] = leaf_id
                self.leaf_next += 1
            assignments[m] = self.leaf_map[leaf_gid]

        self.prev_path = {m: paths_global[m].copy() for m in range(self.nParticles)}
        if return_paths:
            return assignments, paths_global
        return assignments


# ============================================================
# HOLMES: FULL INFERENCE LOOP (with lesion switches)
# ============================================================

def full_hier_inference_loop(
    nTrials, nParticles, nFeatures, alpha, omega, f,
    max_depth=10, max_children=5, random_seed=None,
    outcome_idx=2, outcome_idx_per_trial=None,
    include_outcome_in_weight=True,
    length_normalize=False, tau=1.0,
    feedback_mask=None,
    stickiness=None,          # <-- can set to something other than omega if yoyy want to decouple
    use_stopping=True,        # <-- lesion switch
    n_max_causes=None,        # <-- optional override for the cause budget
    return_rprob=False,
):
    """Run HOLMES with optional lesions (stopping / stickiness)."""
    rng = np.random.RandomState(random_seed)
    aPrior = bPrior = omega
    if stickiness is None:
        stickiness = omega       # reproduce the original coupling by default
    eps = 1e-12

    # Cause budget: large enough for the deepest (no-stopping) case, memory-bounded.
    if n_max_causes is None:
        theoretical_leaves = max_children ** max_depth     
        practical_cap = nParticles * nTrials
        n_max_causes = int(min(theoretical_leaves, practical_cap)) + nTrials * max_depth + 8
    nMaxCauses = int(n_max_causes)

    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + aPrior
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + bPrior

    wt = np.ones(nParticles) / nParticles
    rt = np.ones(nParticles) / nParticles
    rProb = np.zeros(nParticles)
    rProb_parts = np.zeros((nParticles, nTrials))

    cEst = np.zeros((nMaxCauses, nTrials))
    rEst = np.zeros(nTrials)
    phiEst = np.zeros((nFeatures, nTrials))
    paths = -np.ones((nParticles, nTrials, max_depth), dtype=int)
    cEst_levels = np.zeros((max_depth, nMaxCauses, nTrials))
    all_idx = np.arange(nFeatures)

    # ---- feedback mask ----
    if feedback_mask is None:
        use_fb = np.ones(nTrials, dtype=bool)
        feat_mask = np.ones((nFeatures, nTrials), dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask)
        if feedback_mask.ndim == 1:
            use_fb = feedback_mask.astype(bool)
            feat_mask = np.tile(use_fb, (nFeatures, 1))
        elif feedback_mask.ndim == 2:
            feat_mask = feedback_mask.astype(bool)
            use_fb = np.any(feat_mask, axis=0)
        else:
            raise ValueError("feedback_mask must be None, (T,) or (nFeatures, T)")

    prior = MHLCMCRP(
        alpha=alpha, max_depth=max_depth, nParticles=nParticles,
        nMaxCauses=nMaxCauses, random_seed=random_seed,
        depth_decay=alpha, max_children=max_children,
        stickiness=stickiness, use_stopping=use_stopping,
    )

    weight_hist = []
    for t in range(nTrials):
        obs = f[:, t].astype(bool)
        has_fb = bool(use_fb[t])
        obs_mask_vec = feat_mask[:, t]
        oi = (outcome_idx if outcome_idx_per_trial is None
              else int(outcome_idx_per_trial[t]))
        non_o = all_idx[all_idx != oi]

        assigns, path_gids = prior.generate_prior(return_paths=True)
        particles[:, t] = assigns
        paths[:, t, :] = path_gids
        for m, k in enumerate(assigns):
            nC[m, k] += 1

        for l in range(nParticles):
            k = particles[l, t]
            n_o = nFC[oi, k, l]
            b_o = bFC[oi, k, l]
            rProb[l] = n_o / max(n_o + b_o, eps)

        log_rt = np.zeros(nParticles)
        log_wt = np.zeros(nParticles)
        for l in range(nParticles):
            k = particles[l, t]
            log_rt[l] = _log_likelihood_sum(
                obs, nFC, bFC, cause_k=k, part_l=l, feat_idx=non_o,
                length_normalize=length_normalize, tau=tau, obs_mask=obs_mask_vec,
            )
            feat_w = all_idx if (has_fb and include_outcome_in_weight) else non_o
            log_wt[l] = _log_likelihood_sum(
                obs, nFC, bFC, cause_k=k, part_l=l, feat_idx=feat_w,
                length_normalize=length_normalize, tau=tau, obs_mask=obs_mask_vec,
            )
            if has_fb:
                nFC[:, k, l] += obs.astype(int) * obs_mask_vec.astype(int)
                bFC[:, k, l] += (1 - obs).astype(int) * obs_mask_vec.astype(int)
            else:
                nFC[non_o, k, l] += obs[non_o].astype(int)
                bFC[non_o, k, l] += (~obs[non_o]).astype(int)

        max_log_wt = np.max(log_wt)
        wt = np.exp(log_wt - max_log_wt)
        wt = wt / np.maximum(wt.sum(), eps)
        max_log_rt = np.max(log_rt)
        rt = np.exp(log_rt - max_log_rt)
        rt = rt / np.maximum(rt.sum(), eps)
        if not np.all(np.isfinite(wt)) or not np.all(wt >= 0):
            wt = np.ones(nParticles) / nParticles
            rt = np.ones(nParticles) / nParticles

        if return_rprob:
            rProb_parts[:, t] = rProb.copy()
        rEst[t] = float(np.dot(rt, rProb))
        cEst[:, t] = weighted_hist(particles[:, t], wt, nMaxCauses)

        for L in range(max_depth):
            valid = paths[:, t, L] >= 0
            if np.any(valid):
                cEst_levels[L, :, t] = weighted_hist(
                    paths[valid, t, L], wt[valid], nMaxCauses
                )

        for i in range(nFeatures):
            ass = particles[:, t]
            n_curr = nFC[i, ass, np.arange(nParticles)]
            b_curr = bFC[i, ass, np.arange(nParticles)]
            phiEst[i, t] = float(np.dot(wt, n_curr / np.maximum(n_curr + b_curr, eps)))

        weight_hist.append(wt.copy())

        idx = rng.choice(nParticles, nParticles, p=wt)
        particles = particles[idx]
        nC = nC[idx]
        nFC = nFC[:, :, idx]
        bFC = bFC[:, :, idx]
        paths = paths[idx]
        prior.resample(idx)

    # ---- reindex paths to dense ids per level ----
    level_gid_to_dense = []
    for L in range(max_depth):
        used = paths[:, :, L].flatten()
        used = used[used >= 0]
        unique = sorted(set(used))
        level_gid_to_dense.append({g: i for i, g in enumerate(unique)})

    paths_dense = -np.ones_like(paths)
    for m in range(nParticles):
        for t in range(nTrials):
            for L in range(max_depth):
                g = paths[m, t, L]
                if g >= 0 and g in level_gid_to_dense[L]:
                    paths_dense[m, t, L] = level_gid_to_dense[L][g]

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
        return (particles, cEst, rEst, phiEst, cEst_levels_dense,
                paths_dense, weight_hist, rProb_parts)
    return particles, cEst, rEst, phiEst, cEst_levels_dense, paths_dense, weight_hist


# ============================================================
# TASK: SHAPE x MANY-COLORS TRANSFER
# ============================================================

@dataclass
class ShapeColorConfig:
    n_colors: int = 6
    n_train_colors: int = 4
    n_test_colors: int = 2
    train_reps_per_stimulus: int = 6
    test_reps_per_stimulus: int = 1
    n_shape_copies: int = 3
    n_color_copies: int = 1
    n_nuisance_features: int = 0
    blocked_train_by_shape: bool = True
    shuffle_within_train_blocks: bool = True
    shuffle_test: bool = True
    reward_rule: str = "shape0"  # or "random_balanced"


def generate_shape_color_transfer_task(seed: int, cfg: ShapeColorConfig):
    rng = np.random.default_rng(seed)
    assert cfg.n_train_colors + cfg.n_test_colors <= cfg.n_colors

    n_shapes = 2
    train_colors = list(range(cfg.n_train_colors))
    test_colors = list(range(cfg.n_train_colors, cfg.n_train_colors + cfg.n_test_colors))

    n_shape_feats = cfg.n_shape_copies * n_shapes
    n_color_feats = cfg.n_color_copies * cfg.n_colors
    n_nonoutcome = n_shape_feats + n_color_feats + cfg.n_nuisance_features
    outcome_idx = n_nonoutcome

    if cfg.reward_rule == "shape0":
        rewarded_shape = 0
    elif cfg.reward_rule == "random_balanced":
        rewarded_shape = int(rng.integers(0, 2))
    else:
        raise ValueError("!!! reward_rule must be 'shape0' or 'random_balanced'")

    nuisance_by_color = {}
    if cfg.n_nuisance_features > 0:
        for c in range(cfg.n_colors):
            nuisance_by_color[c] = rng.integers(0, 2, size=cfg.n_nuisance_features, dtype=int)

    def make_features(shape: int, color: int) -> np.ndarray:
        x = np.zeros(n_nonoutcome, dtype=int)
        for copy in range(cfg.n_shape_copies):
            x[copy * n_shapes + shape] = 1
        color_start = n_shape_feats
        for copy in range(cfg.n_color_copies):
            x[color_start + copy * cfg.n_colors + color] = 1
        if cfg.n_nuisance_features > 0:
            x[n_shape_feats + n_color_feats:] = nuisance_by_color[color]
        return x

    def outcome(shape: int) -> int:
        return int(shape == rewarded_shape)

    train_trials: List[Dict] = []
    for shape in [0, 1]:
        for color in train_colors:
            for rep in range(cfg.train_reps_per_stimulus):
                train_trials.append({
                    "phase": "train", "shape": shape, "color": color,
                    "is_heldout_color": False, "is_first_heldout": False,
                    "reward": outcome(shape), "features": make_features(shape, color),
                })

    if cfg.blocked_train_by_shape:
        blocks = []
        for shape in [0, 1]:
            block = [tr for tr in train_trials if tr["shape"] == shape]
            if cfg.shuffle_within_train_blocks:
                rng.shuffle(block)
            blocks.append(block)
        train_trials = (blocks[0] + blocks[1]) if rng.random() < 0.5 else (blocks[1] + blocks[0])
    else:
        rng.shuffle(train_trials)

    test_trials: List[Dict] = []
    for shape in [0, 1]:
        for color in test_colors:
            for rep in range(cfg.test_reps_per_stimulus):
                test_trials.append({
                    "phase": "test", "shape": shape, "color": color,
                    "is_heldout_color": True, "is_first_heldout": rep == 0,
                    "reward": outcome(shape), "features": make_features(shape, color),
                })
    if cfg.shuffle_test:
        rng.shuffle(test_trials)

    trials = train_trials + test_trials
    F = np.zeros((n_nonoutcome + 1, len(trials)), dtype=int)
    for t, tr in enumerate(trials):
        F[:n_nonoutcome, t] = tr["features"]
        F[outcome_idx, t] = tr["reward"]

    feedback_mask = np.ones_like(F, dtype=bool)
    meta = {
        "task_type": "shape_many_colors_transfer", "seed": seed, "config": asdict(cfg),
        "n_trials": len(trials), "n_train_trials": len(train_trials),
        "n_test_trials": len(test_trials), "outcome_idx": outcome_idx,
        "n_nonoutcome": n_nonoutcome, "n_shape_feats": n_shape_feats,
        "n_color_feats": n_color_feats, "rewarded_shape": rewarded_shape,
        "train_colors": train_colors, "test_colors": test_colors,
        "trials": [{k: v for k, v in tr.items() if k != "features"} for tr in trials],
    }
    return F, meta, feedback_mask


# ============================================================
# METRICS
# ============================================================

def majority_id(assignments: np.ndarray) -> int:
    valid = np.asarray(assignments)
    valid = valid[valid >= 0]
    if valid.size == 0:
        return -1
    vals, counts = np.unique(valid.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def get_flat_assignments(particles: np.ndarray) -> np.ndarray:
    return np.array([majority_id(particles[:, t]) for t in range(particles.shape[1])], dtype=int)


def get_hier_level_assignments(paths: np.ndarray, level: int) -> np.ndarray:
    return np.array([majority_id(paths[:, t, level]) for t in range(paths.shape[1])], dtype=int)


def count_hier_nodes(paths: np.ndarray) -> Dict[str, int]:
    out = {}
    for L in range(paths.shape[2]):
        vals = paths[:, :, L].ravel()
        vals = vals[vals >= 0]
        if vals.size:
            out[f"level_{L}"] = int(np.unique(vals).size)
    return out


def acc(prob, y, mask) -> float:
    if not np.any(mask):
        return np.nan
    return float(np.mean((prob[mask] > 0.5).astype(int) == y[mask]))


def brier_score(prob, y, mask) -> float:
    if not np.any(mask):
        return np.nan
    p = np.asarray(prob[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    return float(np.mean((p - yy) ** 2))


def log_loss_score(prob, y, mask, eps=1e-9) -> float:
    if not np.any(mask):
        return np.nan
    p = np.clip(np.asarray(prob[mask], dtype=float), eps, 1.0 - eps)
    yy = np.asarray(y[mask], dtype=float)
    return float(np.mean(-(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p))))


def shape_probability_summary(prob, shape, mask) -> Dict[str, float]:
    out = {}
    for s in [0, 1]:
        smask = mask & (shape == s)
        out[f"shape{s}_mean_prob"] = float(np.mean(prob[smask])) if np.any(smask) else np.nan
    out["shape_contrast"] = out["shape0_mean_prob"] - out["shape1_mean_prob"]
    return out


def pairwise_same_group_same_node(assignments, group, mask) -> Dict[str, float]:
    idx = np.where(mask & (assignments >= 0))[0]
    if len(idx) < 2:
        return {"same_group_same_node": np.nan, "diff_group_same_node": np.nan, "separation": np.nan}
    same_total = same_hit = diff_total = diff_hit = 0
    max_pairs, pairs = 30000, 0
    for ai in range(len(idx)):
        for bi in range(ai + 1, len(idx)):
            a, b = idx[ai], idx[bi]
            same_group = group[a] == group[b]
            same_node = assignments[a] == assignments[b]
            if same_group:
                same_total += 1; same_hit += int(same_node)
            else:
                diff_total += 1; diff_hit += int(same_node)
            pairs += 1
            if pairs >= max_pairs:
                break
        if pairs >= max_pairs:
            break
    same = float(same_hit / same_total) if same_total else np.nan
    diff = float(diff_hit / diff_total) if diff_total else np.nan
    sep = same - diff if np.isfinite(same) and np.isfinite(diff) else np.nan
    return {"same_group_same_node": same, "diff_group_same_node": diff, "separation": sep}


def choose_best_hier_level_for_shape(paths, shapes, train_mask):
    best_L, best_sep, best_diag = -1, -np.inf, {}
    diag_by_level = {}
    for L in range(paths.shape[2]):
        if not np.any(paths[:, :, L] >= 0):
            continue
        assign = get_hier_level_assignments(paths, L)
        diag = pairwise_same_group_same_node(assign, shapes, train_mask)
        diag_by_level[f"level_{L}"] = diag
        if np.isfinite(diag["separation"]) and diag["separation"] > best_sep:
            best_L, best_sep, best_diag = L, diag["separation"], diag
    return best_L, {"best_diag": best_diag, "diag_by_level": diag_by_level}


# ============================================================
# MODEL VARIANTS
# ============================================================
# family: "flat" or "hier".
# For hier variants: use_stopping / stickiness_mode control the lesion.
#   stickiness_mode "omega" -> use omega (original coupling)
#                   "off"   -> 0.0
# For flat variants: flat_stickiness_mode "off" or "on".

VARIANTS = {
    "flat":               {"family": "flat", "flat_stick": "off"},
    "flat_sticky":        {"family": "flat", "flat_stick": "on"},   # enhancement
    "holmes_full":        {"family": "hier", "use_stopping": True,  "stick": "on"},
    "holmes_no_stick":    {"family": "hier", "use_stopping": True,  "stick": "off"},
    "holmes_no_stop":     {"family": "hier", "use_stopping": False, "stick": "on"},

}

# Ordered for display
VARIANT_ORDER = ["flat", "flat_sticky",
                 "holmes_no_stop", "holmes_no_stick", "holmes_full"]


def _run_flat_variant(spec, F, oi, fb, seed, n_particles, alpha, omega,
                      length_normalize, tau, flat_stickiness):
    stick = flat_stickiness if spec["flat_stick"] == "on" else 0.0
    p_flat, _, r_flat, _ = one_layer_inference_loop(
        nTrials=F.shape[1], nParticles=n_particles, nFeatures=F.shape[0],
        regime=omega, alpha=alpha, f=F,
        outcome_idx_per_trial=np.full(F.shape[1], oi, dtype=int),
        feedback_mask=fb[oi, :], random_seed=seed,
        length_normalize=length_normalize, tau=tau, stickiness=stick,
    )
    return r_flat, p_flat


def _run_hier_variant(spec, F, oi, fb, seed, n_particles, alpha, omega,
                      max_depth, max_children, length_normalize, tau, hier_stickiness):
    stick = hier_stickiness if spec["stick"] == "on" else 0.0
    out = full_hier_inference_loop(
        nTrials=F.shape[1], nParticles=n_particles, nFeatures=F.shape[0],
        alpha=alpha, omega=omega, f=F, max_depth=max_depth, max_children=max_children,
        outcome_idx=oi, feedback_mask=fb, random_seed=seed,
        length_normalize=length_normalize, tau=tau,
        stickiness=stick, use_stopping=spec["use_stopping"],
    )
    p_hier, _, r_hier, _, _, paths_hier, _ = out
    return r_hier, paths_hier


def run_one_seed(seed, cfg, n_particles, alpha, omega, max_depth, max_children,
                 length_normalize, tau, hier_stickiness, flat_stickiness):
    F, meta, fb = generate_shape_color_transfer_task(seed, cfg)
    y = F[meta["outcome_idx"], :].astype(int)
    oi = int(meta["outcome_idx"])
    trials = meta["trials"]
    phase = np.array([tr["phase"] for tr in trials])
    shape = np.array([tr["shape"] for tr in trials], dtype=int)
    color = np.array([tr["color"] for tr in trials], dtype=int)
    is_first = np.array([tr["is_first_heldout"] for tr in trials], dtype=bool)
    train_mask = phase == "train"
    first_heldout_mask = (phase == "test") & is_first
    rewarded_shape = int(meta["rewarded_shape"])
    sign = 1 if rewarded_shape == 0 else -1  # sign-correct for random_balanced

    rows = []
    for name in VARIANT_ORDER:
        spec = VARIANTS[name]
        if spec["family"] == "flat":
            r, particles = _run_flat_variant(
                spec, F, oi, fb, seed, n_particles, alpha, omega,
                length_normalize, tau, flat_stickiness)
            assign = get_flat_assignments(particles)
            shape_diag = pairwise_same_group_same_node(assign, shape, train_mask)
            color_diag = pairwise_same_group_same_node(assign, color, train_mask)
            struct_level = -1
            n_units = int(len(np.unique(assign[assign >= 0])))
            nodes_desc = f"clusters={n_units}"
            shape_sep = shape_diag["separation"]
            color_sep = color_diag["separation"]
        else:
            r, paths = _run_hier_variant(
                spec, F, oi, fb, seed, n_particles, alpha, omega,
                max_depth, max_children, length_normalize, tau, hier_stickiness)
            best_L, hier_info = choose_best_hier_level_for_shape(paths, shape, train_mask)
            best_assign = (get_hier_level_assignments(paths, best_L)
                           if best_L >= 0 else np.full(F.shape[1], -1))
            color_diag = pairwise_same_group_same_node(best_assign, color, train_mask)
            struct_level = int(best_L)
            nodes = count_hier_nodes(paths)
            n_units = int(sum(nodes.values()))
            nodes_desc = ";".join(f"{k}:{v}" for k, v in nodes.items())
            shape_sep = float(hier_info["best_diag"].get("separation", np.nan))
            color_sep = color_diag["separation"]

        held = shape_probability_summary(r, shape, first_heldout_mask)

        rows.append({
            "seed": seed,
            "variant": name,
            "family": spec["family"],
            "rewarded_shape": rewarded_shape,
            "train_acc": acc(r, y, train_mask),
            "first_heldout_acc": acc(r, y, first_heldout_mask),
            "heldout_brier": brier_score(r, y, first_heldout_mask),
            "heldout_logloss": log_loss_score(r, y, first_heldout_mask),
            "heldout_shape_contrast": held["shape_contrast"],
            "heldout_signed_contrast": sign * held["shape_contrast"],
            "shape_separation": shape_sep,
            "color_separation": color_sep,
            "best_shape_level": struct_level,
            "n_units": n_units,
            "nodes_desc": nodes_desc,
        })
    return pd.DataFrame(rows)


# ============================================================
# COMPARSONS
# ============================================================

def _paired(df, a, b, metric):
    """Paired difference (a - b) across seeds, with a paired t-test if possible."""
    pa = df[df.variant == a].set_index("seed")[metric]
    pb = df[df.variant == b].set_index("seed")[metric]
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common] - pb.loc[common]).dropna()
    mean = float(d.mean()) if len(d) else np.nan
    sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    t = p = np.nan
    if len(d) > 1 and sd and np.isfinite(sd) and sd > 0:
        try:
            from scipy import stats as _stats
            t, p = _stats.ttest_rel(pa.loc[common], pb.loc[common])
            t, p = float(t), float(p)
        except Exception:
            pass
    return mean, sd, t, p, len(d)


def _fmt_contrast(label, df, a, b, metric, higher_better=True):
    mean, sd, t, p, n = _paired(df, a, b, metric)
    arrow = "" if not np.isfinite(mean) else ("(+)" if (mean > 0) == higher_better else "(-)")
    sd_s = f"{sd:.4f}" if np.isfinite(sd) else "  nan"
    p_s = f"{p:.4f}" if np.isfinite(p) else "  nan"
    t_s = f"{t:+.3f}" if np.isfinite(t) else "  nan"
    return f"  {label:<34} {a} - {b}: {mean:+.4f} +/- {sd_s}  (t={t_s}, p={p_s}, n={n}) {arrow}"


def report(df: pd.DataFrame):
    print("\n" + "=" * 92)
    print("PER-VARIANT MEANS ACROSS SEEDS (held-out = first exposure to each novel color)")
    print("=" * 92)
    agg = (df.groupby("variant")
             .agg(train_acc=("train_acc", "mean"),
                  first_heldout_acc=("first_heldout_acc", "mean"),
                  heldout_brier=("heldout_brier", "mean"),
                  heldout_logloss=("heldout_logloss", "mean"),
                  heldout_signed_contrast=("heldout_signed_contrast", "mean"),
                  shape_separation=("shape_separation", "mean"),
                  n_units=("n_units", "mean"))
             .reindex(VARIANT_ORDER))
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(agg.to_string())

    print("\n" + "=" * 92)
    print("KEY CONTRASTS (paired across seeds)")
    print("=" * 92)

    print("\n[Q1] Is the COMBINATION needed, or does the structure alone suffice?")
    print(_fmt_contrast("full vs structure-only",
                        df, "holmes_full", "holmes_struct_only",
                        "first_heldout_acc", higher_better=True))
    print(_fmt_contrast("full vs structure-only (Brier)",
                        df, "holmes_full", "holmes_struct_only",
                        "heldout_brier", higher_better=False))
    print(_fmt_contrast("full vs structure-only (signed)",
                        df, "holmes_full", "holmes_struct_only",
                        "heldout_signed_contrast", higher_better=True))

    print("\n[Q2] Marginal value of each mechanism (drop-one from full):")
    print(_fmt_contrast("value of stickiness",
                        df, "holmes_full", "holmes_no_stick",
                        "first_heldout_acc", higher_better=True))
    print(_fmt_contrast("value of stopping",
                        df, "holmes_full", "holmes_no_stop",
                        "first_heldout_acc", higher_better=True))

    print("\n[Q3] Flat-model enhancement (add stickiness to the flat CRP):")
    print(_fmt_contrast("sticky-flat vs flat",
                        df, "flat_sticky", "flat",
                        "first_heldout_acc", higher_better=True))
    print(_fmt_contrast("sticky-flat vs flat (Brier)",
                        df, "flat_sticky", "flat",
                        "heldout_brier", higher_better=False))

    print("\n[Q4] Does sticky-flat close the gap to full HOLMES?")
    print(_fmt_contrast("HOLMES-full vs sticky-flat",
                        df, "holmes_full", "flat_sticky",
                        "first_heldout_acc", higher_better=True))
    print(_fmt_contrast("HOLMES-full vs flat",
                        df, "holmes_full", "flat",
                        "first_heldout_acc", higher_better=True))
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HOLMES ablation + flat enhancement suite")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_particles", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.5,
                        help="Bernoulli prior pseudocount (aPrior=bPrior).")
    parser.add_argument("--hier_stickiness", type=float, default=None,
                        help="HOLMES stickiness strength; default None => use omega "
                             "(reproduces original coupling).")
    parser.add_argument("--flat_stickiness", type=float, default=None,
                        help="Flat-model stickiness strength; default None => match hier.")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--max_children", type=int, default=5)
    parser.add_argument("--no_length_normalize", action="store_true")
    parser.add_argument("--tau", type=float, default=1.0)
    # task config
    parser.add_argument("--n_colors", type=int, default=6)
    parser.add_argument("--n_train_colors", type=int, default=4)
    parser.add_argument("--n_test_colors", type=int, default=2)
    parser.add_argument("--train_reps", type=int, default=6)
    parser.add_argument("--test_reps", type=int, default=1)
    parser.add_argument("--shape_copies", type=int, default=3)
    parser.add_argument("--color_copies", type=int, default=1)
    parser.add_argument("--nuisance_features", type=int, default=0)
    parser.add_argument("--shuffled_train", action="store_true")
    parser.add_argument("--reward_rule", type=str, default="shape0",
                        choices=["shape0", "random_balanced"])
    parser.add_argument("--out_prefix", type=str, default="holmes_ablation")
    args = parser.parse_args()

    # Resolve stickiness defaults
    hier_stick = args.omega if args.hier_stickiness is None else args.hier_stickiness
    flat_stick = hier_stick if args.flat_stickiness is None else args.flat_stickiness
    length_normalize = not args.no_length_normalize

    cfg = ShapeColorConfig(
        n_colors=args.n_colors, n_train_colors=args.n_train_colors,
        n_test_colors=args.n_test_colors,
        train_reps_per_stimulus=args.train_reps, test_reps_per_stimulus=args.test_reps,
        n_shape_copies=args.shape_copies, n_color_copies=args.color_copies,
        n_nuisance_features=args.nuisance_features,
        blocked_train_by_shape=not args.shuffled_train, reward_rule=args.reward_rule,
    )

    print("=" * 92)
    print("HOLMES ABLATION + FLAT ENHANCEMENT")
    print("=" * 92)
    print(f"Variants: {VARIANT_ORDER}")
    print(f"Seeds: {args.seeds} | particles={args.n_particles} | alpha={args.alpha} "
          f"| omega={args.omega}")
    print(f"hier_stickiness={hier_stick} | flat_stickiness={flat_stick} "
          f"| length_normalize={length_normalize} | tau={args.tau}")
    print(f"max_depth={args.max_depth} | max_children={args.max_children}")
    print(f"Task: {cfg}")

    all_rows = []
    for seed in args.seeds:
        print(f"\nRunning seed {seed} ...")
        df_seed = run_one_seed(
            seed, cfg, args.n_particles, args.alpha, args.omega,
            args.max_depth, args.max_children, length_normalize, args.tau,
            hier_stick, flat_stick,
        )
        all_rows.append(df_seed)
        # brief per-seed line
        piv = df_seed.set_index("variant")["first_heldout_acc"]
        summary = " | ".join(f"{v}={piv.get(v, np.nan):.2f}" for v in VARIANT_ORDER)
        print(f"  first-heldout acc: {summary}")

    df = pd.concat(all_rows, ignore_index=True)
    report(df)

    # Save
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = Path(args.out_prefix).name
    csv_path = f"{prefix}_{ts}_long.csv"
    df.to_csv(csv_path, index=False)

    agg = (df.groupby("variant").mean(numeric_only=True).reindex(VARIANT_ORDER))
    agg_path = f"{prefix}_{ts}_summary.csv"
    agg.to_csv(agg_path)

    meta = {"args": vars(args), "config": asdict(cfg),
            "hier_stickiness": hier_stick, "flat_stickiness": flat_stick,
            "variants": VARIANTS}
    json_path = f"{prefix}_{ts}_meta.json"
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved per-(seed,variant) rows -> {csv_path}")
    print(f"Saved variant summary         -> {agg_path}")
    print(f"Saved run metadata            -> {json_path}")


if __name__ == "__main__":
    main()