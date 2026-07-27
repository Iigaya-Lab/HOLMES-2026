#!/usr/bin/env python3
"""
 label-generalization (transfer) probe — ablations!

This is a faster, fairness-hardened rebuild of the transfer_task probe.

Given a latent factor (e.g. species) that
partitions trials into categories, we label a few "anchor" trials, read off the
cluster/node the model assigned them, and predict "same category" for every trial
sharing that cluster and "different" otherwise. Scoring those same/different calls
against the true factor (accuracy + F1) measures whether the model's learned
partition carves the space at the grain of that factor.

The flat model has one partition, so all three readouts reduce to scoring that
single partition (held_out still evaluates on the held-out split for fairness).

MODELS / LESIONS
----------------
HOLMES exposes the same two lesion
switches as the shape/color ablation: `use_stopping` and a decoupled `stickiness`.
Variants: flat, flat_sticky, holmes_full, holmes_no_stick, holmes_no_stop,
holmes_struct_only (again, not super relevant)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ============================================================
# SHARED LIKELIHOOD HELPERS
# ============================================================

def weighted_hist(labels, weights, K):
    out = np.zeros(K, dtype=float)
    for lab, w in zip(labels, weights):
        lab = int(lab)
        if 0 <= lab < K:
            out[lab] += w
    return out


def _bernoulli_predictive(obs_bool_vec, n_vec, b_vec, eps=1e-12):
    denom = np.maximum(n_vec + b_vec, eps)
    O = obs_bool_vec.astype(np.float64)
    return O * (n_vec / denom) + (1.0 - O) * (b_vec / denom)


def _likelihood_product(obs_bool, nFC, bFC, k, p, feat_idx, length_normalize=False, tau=1.0, eps=1e-12):
    if len(feat_idx) == 0:
        return 1.0
    preds = _bernoulli_predictive(obs_bool[feat_idx], nFC[feat_idx, k, p], bFC[feat_idx, k, p], eps=eps)
    preds = np.maximum(preds, eps)
    if length_normalize:
        return float(np.exp(tau * np.mean(np.log(preds))))
    return float(np.prod(preds))


# ============================================================
# FLAT MODEL (one-layer CRP) with optional stickiness
# ============================================================

class StickyCRP:
    def __init__(self, alpha, nParticles, stickiness=0.0, rng=None):
        self.alpha = float(alpha)
        self.stickiness = float(stickiness)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.prev = -np.ones(int(nParticles), dtype=int)

    def generate_prior(self, nKTminus1):
        M, Kmax = nKTminus1.shape
        assignmentsT = np.zeros(M, dtype=int)
        nKT = nKTminus1.copy()
        for m in range(M):
            K = int(np.sum(nKTminus1[m] > 0))
            probs = np.append(nKTminus1[m, :K], self.alpha)
            probs = np.append(probs, np.zeros(Kmax - K - 1))
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
        self.prev = self.prev[idx]


def one_layer_inference_loop(
    nTrials, nParticles, nFeatures, regime, alpha, f,
    outcome_idx_per_trial, feedback_mask=None,
    include_outcome_in_weight=True, length_normalize=False, tau=1.0,
    random_seed=0, stickiness=0.0,
):
    """Flat CRP model. stickiness=0 reproduces the original flat_model exactly."""
    rng = np.random.default_rng(random_seed)
    if feedback_mask is None:
        feedback_mask = np.ones(nTrials, dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask, dtype=bool)

    # NOTE: the flat model keeps the ORIGINAL per-particle product-space math so that
    # stickiness=0 reproduces flat_model.one_layer_inference_loop bit-for-bit (the
    # resampling RNG is sensitive to last-bit differences in the weights). Flat is
    # cheap (small arrays), so it is not vectorized.
    nMaxCauses = nTrials
    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + regime

    wt = np.ones(nParticles) / nParticles
    rt = np.ones(nParticles) / nParticles
    rProb = np.zeros(nParticles)
    rEst = np.zeros(nTrials)
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
            rt[p] = _likelihood_product(obs, nFC, bFC, k, p, non_outcome_dims, length_normalize, tau)
            n_o = nFC[outcome_dim, k, p]; b_o = bFC[outcome_dim, k, p]
            rProb[p] = n_o / max(n_o + b_o, 1e-12)
            feat_idx = all_idx if (has_fb and include_outcome_in_weight) else non_outcome_dims
            wt[p] = _likelihood_product(obs, nFC, bFC, k, p, feat_idx, length_normalize, tau)
            if has_fb:
                nFC[:, k, p] += obs
                bFC[:, k, p] += ~obs
            else:
                nFC[non_outcome_dims, k, p] += obs[non_outcome_dims]
                bFC[non_outcome_dims, k, p] += ~obs[non_outcome_dims]

        wt /= max(wt.sum(), 1e-12)
        rt /= max(rt.sum(), 1e-12)
        rEst[t] = float(np.dot(rt, rProb))

        idx = rng.choice(nParticles, nParticles, p=wt)
        particles = particles[idx]
        nC = nC[idx]
        nFC = nFC[:, :, idx]
        bFC = bFC[:, :, idx]
        crp.resample(idx)

    return particles, rEst


# ============================================================
# HOLMES: registry, tree, prior (with lesion switches)
# ============================================================

class GlobalNodeRegistry:
    def __init__(self, max_children=5):
        self.max_children = max_children
        self.registry = {}; self.children = {}; self.next_id = 1
        self.registry[(0, None, 0)] = 0; self.children[None] = [0]; self.children[0] = []

    def get_or_create(self, level, parent_id, branch_index):
        sig = (level, parent_id, branch_index)
        if sig in self.registry:
            return self.registry[sig]
        gid = self.next_id; self.registry[sig] = gid; self.next_id += 1
        self.children.setdefault(parent_id, []).append(gid); self.children[gid] = []
        return gid


class NCRPTreeParticle:
    def __init__(self, alpha, max_depth, max_children):
        self.alpha = float(alpha); self.max_depth = int(max_depth)
        self.max_children = max_children
        self.level_counts = [[] for _ in range(max_depth)]


class MHLCMCRP:
    def __init__(self, alpha, max_depth, nParticles, nMaxCauses, random_seed=None,
                 depth_decay=1.0, max_children=5, stickiness=1.0, use_stopping=True,
                 fixed_depth=None):
        # fixed_depth: when use_stopping=False, descend to this depth instead of the
        # full max_depth. This decouples the "no depth inference" lesion from the arbitrary computational ceiling (max_depth=20)
        self._effective_depth = int(max_depth) if (use_stopping or fixed_depth is None) \
            else int(min(max_depth, fixed_depth))
        self.alpha = float(alpha); self.max_depth = int(max_depth)
        self.nParticles = nParticles; self.nMaxCauses = nMaxCauses
        self.depth_decay = float(depth_decay); self.max_children = max_children
        self.stickiness = float(stickiness); self.use_stopping = bool(use_stopping)
        self.rng = np.random.RandomState(random_seed)
        self.trees = [NCRPTreeParticle(alpha, max_depth, max_children) for _ in range(nParticles)]
        self.leaf_map = dict(); self.leaf_next = 0
        self.global_registry = GlobalNodeRegistry(max_children=max_children)
        self.prev_path = {m: None for m in range(nParticles)}

    def resample(self, idx):
        self.trees = [self.trees[i] for i in idx]
        self.prev_path = {m: self.prev_path[i] for m, i in enumerate(idx)}

    def generate_prior(self, return_paths=False):
        assignments = np.zeros(self.nParticles, dtype=int)
        paths_global = -np.ones((self.nParticles, self.max_depth), dtype=int)
        alpha_L = self.alpha
        for m in range(self.nParticles):
            particle = self.trees[m]; parent_gid = 0; path_gids = []
            prev_path = self.prev_path.get(m, None)
            for L in range(self._effective_depth):
                if self.use_stopping and L > 0:
                    stop_prob = 1.0 / (1.0 + alpha_L)
                    if self.rng.rand() < stop_prob:
                        break
                alpha_L = self.alpha * np.exp(-self.depth_decay * L)
                counts = particle.level_counts[L]; K = len(counts)
                if K == 0:
                    probs = np.array([1.0]); choice = 0
                    particle.level_counts[L].append(1)
                else:
                    if K < self.max_children:
                        probs = np.zeros(K + 1); probs[:K] = np.array(counts, float); probs[K] = alpha_L
                    else:
                        probs = np.array(counts, float)
                    if (self.stickiness > 0.0 and prev_path is not None
                            and L < len(prev_path) and prev_path[L] >= 0):
                        prev_gid = prev_path[L]
                        for existing in range(K):
                            cand = self.global_registry.get_or_create(L, parent_gid, existing)
                            if cand == prev_gid:
                                probs[existing] *= (1.0 + self.stickiness); break
                    probs_norm = probs / np.sum(probs)
                    choice = self.rng.choice(len(probs_norm), p=probs_norm)
                    if choice == K and K < self.max_children:
                        particle.level_counts[L].append(1)
                    else:
                        particle.level_counts[L][choice] += 1
                gid = self.global_registry.get_or_create(L, parent_gid, choice)
                parent_gid = gid; path_gids.append(gid)
            full_path = -np.ones(self.max_depth, dtype=int); full_path[:len(path_gids)] = path_gids
            paths_global[m] = full_path
            leaf_gid = 0 if len(path_gids) == 0 else path_gids[-1]
            if leaf_gid not in self.leaf_map:
                if self.leaf_next >= self.nMaxCauses:
                    raise RuntimeError(f"Exceeded nMaxCauses={self.nMaxCauses}")
                self.leaf_map[leaf_gid] = self.leaf_next; self.leaf_next += 1
            assignments[m] = self.leaf_map[leaf_gid]
        self.prev_path = {m: paths_global[m].copy() for m in range(self.nParticles)}
        if return_paths:
            return assignments, paths_global
        return assignments


def full_hier_inference_loop(
    nTrials, nParticles, nFeatures, alpha, omega, f,
    max_depth=10, max_children=5, random_seed=None,
    outcome_idx=2, outcome_idx_per_trial=None,
    include_outcome_in_weight=True, length_normalize=False, tau=1.0,
    feedback_mask=None, stickiness=None, use_stopping=True, fixed_depth=6,
):
    """
     HOLMES
     Returns (particles, rEst, paths_dense).
    """
    rng = np.random.RandomState(random_seed)
    aPrior = bPrior = omega
    if stickiness is None:
        stickiness = omega
    nMaxCauses = nTrials * max_depth
    eps = 1e-12

    particles = np.zeros((nParticles, nTrials), dtype=int)
    nC = np.zeros((nParticles, nMaxCauses))
    nFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + aPrior
    bFC = np.zeros((nFeatures, nMaxCauses, nParticles)) + bPrior
    rProb = np.zeros(nParticles)
    rEst = np.zeros(nTrials)
    paths = -np.ones((nParticles, nTrials, max_depth), dtype=int)
    all_idx = np.arange(nFeatures)
    par = np.arange(nParticles)

    if feedback_mask is None:
        use_fb = np.ones(nTrials, dtype=bool)
        feat_mask = np.ones((nFeatures, nTrials), dtype=bool)
    else:
        feedback_mask = np.asarray(feedback_mask)
        if feedback_mask.ndim == 1:
            use_fb = feedback_mask.astype(bool); feat_mask = np.tile(use_fb, (nFeatures, 1))
        else:
            feat_mask = feedback_mask.astype(bool); use_fb = np.any(feat_mask, axis=0)

    prior = MHLCMCRP(alpha=alpha, max_depth=max_depth, nParticles=nParticles,
                     nMaxCauses=nMaxCauses, random_seed=random_seed, depth_decay=alpha,
                     max_children=max_children, stickiness=stickiness, use_stopping=use_stopping,
                     fixed_depth=fixed_depth)

    weight_hist = []  # hoisted
    for t in range(nTrials):
        obs = f[:, t].astype(bool)
        has_fb = bool(use_fb[t]); obsm = feat_mask[:, t]
        oi = outcome_idx if outcome_idx_per_trial is None else int(outcome_idx_per_trial[t])
        non_o = all_idx[all_idx != oi]

        assigns, path_gids = prior.generate_prior(return_paths=True)
        particles[:, t] = assigns; paths[:, t, :] = path_gids
        for m, k in enumerate(assigns):
            nC[m, k] += 1
        k_vec = particles[:, t]

        n_o = nFC[oi, k_vec, par]; b_o = bFC[oi, k_vec, par]
        rProb = n_o / np.maximum(n_o + b_o, eps)

        nCol = nFC[:, k_vec, par]; bCol = bFC[:, k_vec, par]
        denom = np.maximum(nCol + bCol, eps)
        O = obs.astype(np.float64)[:, None]
        pred = O * (nCol / denom) + (1.0 - O) * (bCol / denom)
        logpred = np.log(np.maximum(pred, eps))

        rt_mask = np.zeros(nFeatures, bool); rt_mask[non_o] = True; rt_mask &= obsm
        w_mask = obsm.copy() if (has_fb and include_outcome_in_weight) else rt_mask.copy()

        if length_normalize:
            rc = rt_mask.sum(); wc = w_mask.sum()
            log_rt = (tau * (logpred * rt_mask[:, None]).sum(0) / rc) if rc > 0 else np.zeros(nParticles)
            log_wt = (tau * (logpred * w_mask[:, None]).sum(0) / wc) if wc > 0 else np.zeros(nParticles)
        else:
            log_rt = (logpred * rt_mask[:, None]).sum(0)
            log_wt = (logpred * w_mask[:, None]).sum(0)

        if has_fb:
            np.add.at(nFC, (slice(None), k_vec, par),
                      (obs.astype(int) * obsm.astype(int))[:, None] * np.ones((1, nParticles), int))
            np.add.at(bFC, (slice(None), k_vec, par),
                      ((1 - obs).astype(int) * obsm.astype(int))[:, None] * np.ones((1, nParticles), int))
        else:
            an = np.zeros((nFeatures, nParticles), int); ab = np.zeros((nFeatures, nParticles), int)
            an[non_o, :] = obs[non_o].astype(int)[:, None]; ab[non_o, :] = (~obs[non_o]).astype(int)[:, None]
            np.add.at(nFC, (slice(None), k_vec, par), an)
            np.add.at(bFC, (slice(None), k_vec, par), ab)

        m_wt = np.max(log_wt); wt = np.exp(log_wt - m_wt); wt = wt / np.maximum(wt.sum(), eps)
        m_rt = np.max(log_rt); rt = np.exp(log_rt - m_rt); rt = rt / np.maximum(rt.sum(), eps)
        if not np.all(np.isfinite(wt)) or not np.all(wt >= 0):
            wt = np.ones(nParticles) / nParticles; rt = np.ones(nParticles) / nParticles

        rEst[t] = float(np.dot(rt, rProb))
        weight_hist.append(wt.copy())

        idx = rng.choice(nParticles, nParticles, p=wt)
        nu = prior.leaf_next  # only causes 0..nu-1 have been touched; rest frozen at prior
        particles = particles[idx]
        nC = nC[idx]
        nFC[:, :nu, :] = nFC[:, :nu, idx]
        bFC[:, :nu, :] = bFC[:, :nu, idx]
        paths = paths[idx]
        prior.resample(idx)

    # dense reindex of paths (identical to original post-processing)
    level_gid_to_dense = []
    for L in range(max_depth):
        used = paths[:, :, L].flatten(); used = used[used >= 0]
        level_gid_to_dense.append({g: i for i, g in enumerate(sorted(set(used)))})
    paths_dense = -np.ones_like(paths)
    for m in range(nParticles):
        for t in range(nTrials):
            for L in range(max_depth):
                g = paths[m, t, L]
                if g >= 0 and g in level_gid_to_dense[L]:
                    paths_dense[m, t, L] = level_gid_to_dense[L][g]

    return particles, rEst, paths_dense


# ============================================================
# TASK (unchanged generator from transfer_task.py)
# ============================================================

def generate_scalable_hierarchical_task_quiet(n_levels=3, trials_per_context=10, seed=None):
    rng = np.random.default_rng(seed)
    n_top_levels = max(0, n_levels - 2)
    n_species_feats = 2 if n_levels >= 2 else 0
    n_color_feats = 4 if n_levels >= 1 else 0
    total_features = n_top_levels + n_species_feats + n_color_feats
    outcome_idx = total_features

    def generate_all_contexts(nl):
        if nl == 1:
            return [[c] for c in range(4)]
        elif nl == 2:
            return [[color, species] for species in [0, 1] for color in range(4)]
        else:
            sub = generate_all_contexts(nl - 1)
            return [s + [top] for top in [0, 1] for s in sub]

    all_contexts = generate_all_contexts(n_levels)
    n_contexts = len(all_contexts)
    n_trials = n_contexts * trials_per_context

    prototypes, outcomes = {}, {}
    for ctx_id, context in enumerate(all_contexts):
        features = [0] * total_features
        for i in range(n_top_levels):
            features[i] = context[n_levels - i - 1]
        if n_levels >= 2:
            features[n_top_levels] = context[1]; features[n_top_levels + 1] = context[1]
        if n_levels >= 1:
            features[n_top_levels + n_species_feats + context[0]] = 1
        if n_levels <= 1:
            outcome = 0
        elif n_levels == 2:
            outcome = 1 if context[1] == 0 else 0
        else:
            outcome = 1 if (context[n_levels - 1] == 0 and context[n_levels - 2] == 0) else 0
        prototypes[ctx_id] = features; outcomes[ctx_id] = outcome

    all_trials = []
    for ctx_id in range(n_contexts):
        for _ in range(trials_per_context):
            features = prototypes[ctx_id].copy()
            if rng.random() < 0.02:
                cfi = rng.choice(range(n_top_levels + n_species_feats,
                                       n_top_levels + n_species_feats + n_color_feats))
                features[cfi] = 1 - features[cfi]
            all_trials.append({'context_id': ctx_id, 'features': features,
                               'outcome': outcomes[ctx_id], 'context': all_contexts[ctx_id]})
    rng.shuffle(all_trials)

    F = np.zeros((total_features + 1, n_trials), dtype=int)
    true_factors = {i: [] for i in range(1, n_levels + 1)}
    for t, trial in enumerate(all_trials):
        F[:total_features, t] = trial['features']; F[outcome_idx, t] = trial['outcome']
        for level in range(1, n_levels + 1):
            true_factors[level].append(trial['context'][level - 1])
    true_factors_dict = {f'level_{lv}': np.array(v) for lv, v in true_factors.items()}
    feedback_mask = np.ones((total_features + 1, n_trials), dtype=bool)
    meta = {'n_levels': n_levels, 'n_contexts': n_contexts, 'true_factors': true_factors_dict,
            'outcome_idx': outcome_idx, 'n_features': total_features}
    return F, meta, feedback_mask


# ============================================================
# READOUT + PROBE 
# ============================================================

    #   * _majority_id: majority node via np.bincount+argmax over valid (>=0)
    #     assignments (ties -> lowest index, matching numpy argmax).
    #   * single labeled anchor = first factor==0 trial.
    #   * FLAT: predict same/diff by whether each trial's majority cluster equals
    #     the anchor's cluster; score on ALL trials.
    #   * HIER: fixed readout level = (test_true_level - 1), walking DOWNWARD
    #     to level 0 and taking the first level with >=10% valid trials; use that
    #     level's majority nodes; score on ALL trials.
    #   * accuracy/precision/recall/F1 with positive_label=0.
    #

def _majority_id(particle_assignments):
    """Majority node id (>=0) or -1."""
    if len(particle_assignments) == 0:
        return -1
    valid = particle_assignments[particle_assignments >= 0]
    if len(valid) == 0:
        return -1
    counts = np.bincount(valid.astype(int))
    return int(np.argmax(counts))


def compute_transfer_metrics_with_f1(predictions, true_labels, positive_label=0):
    """copy of transfer_task.compute_transfer_metrics_with_f1."""
    true_positive_mask = (true_labels == positive_label)
    pred_positive_mask = (predictions == positive_label)
    TP = int(np.sum(pred_positive_mask & true_positive_mask))
    FP = int(np.sum(pred_positive_mask & ~true_positive_mask))
    FN = int(np.sum(~pred_positive_mask & true_positive_mask))
    TN = int(np.sum(~pred_positive_mask & ~true_positive_mask))
    n = len(predictions)
    accuracy = (TP + TN) / n if n > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def probe_flat_exact(particles, true_factor):
    """FLAT transfer readout, exactly as in transfer_task.py. Scores on ALL trials."""
    n_trials = particles.shape[1]
    factor_0 = np.where(true_factor == 0)[0]
    if len(factor_0) == 0:
        return None, -1
    labeled = int(factor_0[0])
    anchor = _majority_id(particles[:, labeled])
    pred = np.array([0 if _majority_id(particles[:, t]) == anchor else 1
                     for t in range(n_trials)])
    return compute_transfer_metrics_with_f1(pred, true_factor, positive_label=0), -1


def probe_hier_exact(paths, true_factor, test_true_level):
    """
    HIER transfer readout, exactly as in transfer_task.py:
    fixed start level = test_true_level-1, walk downward to 0, take the first
    level with >=10% valid trials, score its majority nodes on ALL trials.
    """
    n_trials = paths.shape[1]
    factor_0 = np.where(true_factor == 0)[0]
    if len(factor_0) == 0:
        return None, -1
    labeled = int(factor_0[0])
    start_level = test_true_level - 1
    for try_level in range(start_level, -1, -1):
        if try_level >= paths.shape[2]:
            continue
        nodes = np.array([_majority_id(paths[:, t, try_level]) for t in range(n_trials)])
        valid_frac = np.mean(nodes >= 0)
        if valid_frac >= 0.10:
            anchor = _majority_id(paths[:, labeled, try_level])
            if anchor >= 0:
                pred = np.array([0 if nodes[t] == anchor else 1 for t in range(n_trials)])
                return compute_transfer_metrics_with_f1(pred, true_factor, positive_label=0), try_level
    return None, -1


# ============================================================
# VARIANTS
# ============================================================

VARIANTS = {
    "flat":               {"family": "flat", "flat_stick": "off"},
    "flat_sticky":        {"family": "flat", "flat_stick": "on"},
    "holmes_full":        {"family": "hier", "use_stopping": True,  "stick": "on"},
    "holmes_no_stick":    {"family": "hier", "use_stopping": True,  "stick": "off"},
    "holmes_no_stop":     {"family": "hier", "use_stopping": False, "stick": "on"},
    "holmes_struct_only": {"family": "hier", "use_stopping": False, "stick": "off"},
}
VARIANT_ORDER = ["flat", "flat_sticky", "holmes_struct_only",
                 "holmes_no_stop", "holmes_no_stick", "holmes_full"]


@dataclass
class RunConfig:
    n_particles: int = 200
    alpha: float = 2.0
    omega: float = 0.5
    max_depth: int = 20
    max_children: int = 20
    fixed_depth: int = 10             # descent depth for the no-stopping lesion
    length_normalize: bool = False
    tau: float = 1.0
    hier_stickiness: Optional[float] = None
    flat_stickiness: Optional[float] = None
    trials_per_context: int = 10


def run_ablation_point(alpha, omega, seed, max_levels=5,
                       n_particles=200, max_depth=20, max_children=20,
                       trials_per_context=10, fixed_depth=10,
                       length_normalize=False, tau=1.0,
                       hier_stickiness=None, flat_stickiness=None,
                       variants=None):
    """
    Run all variants at ONE (alpha, omega, seed) cell and return a tidy long
    DataFrame (one row per variant x test_level). This is the atomic primitive:
    call it directly for a single point, or loop it over a grid / seed set and
    pd.concat the results.

    Fidelity: with defaults (hier_stickiness=None -> omega, use_stopping=True),
    the 'holmes_full' rows reproduce the paper's Hier numbers and 'flat' the
    paper's Flat numbers, bit-for-bit, at the given (alpha, omega, seed).

    Parameters mirror the sweep so you can pass a pickle cell's alpha/omega
    straight in. `variants` optionally restricts which of VARIANT_ORDER to run
    (e.g. ["flat", "holmes_full"] to just anchor against the paper cheaply).

    Returns columns:
        alpha, omega, seed, n_levels, test_level, variant, family,
        transfer_acc, transfer_f1, precision, recall, level_used
    """
    hier_stick = omega if hier_stickiness is None else hier_stickiness
    flat_stick = hier_stick if flat_stickiness is None else flat_stickiness
    variants = list(VARIANT_ORDER if variants is None else variants)
    rows = []

    for n_levels in range(2, max_levels + 1):
        F, meta, fb = generate_scalable_hierarchical_task_quiet(
            n_levels=n_levels, trials_per_context=trials_per_context, seed=seed)
        n_trials = F.shape[1]; oi = meta['outcome_idx']

        # fit each requested variant once; reuse across the factors we probe
        fits = {}
        for name in variants:
            spec = VARIANTS[name]
            if spec["family"] == "flat":
                stick = flat_stick if spec["flat_stick"] == "on" else 0.0
                np.random.seed(seed)  # match transfer_task.py's per-fit reseed
                particles, _ = one_layer_inference_loop(
                    nTrials=n_trials, nParticles=n_particles, nFeatures=F.shape[0],
                    regime=omega, alpha=alpha, f=F,
                    outcome_idx_per_trial=np.full(n_trials, oi, int),
                    feedback_mask=fb[oi, :], random_seed=seed,
                    length_normalize=length_normalize, tau=tau, stickiness=stick)
                fits[name] = ("flat", particles)
            else:
                stick = hier_stick if spec["stick"] == "on" else 0.0
                np.random.seed(seed)
                particles, _, paths = full_hier_inference_loop(
                    nTrials=n_trials, nParticles=n_particles, nFeatures=F.shape[0],
                    alpha=alpha, omega=omega, f=F,
                    max_depth=max_depth, max_children=max_children,
                    outcome_idx=oi, feedback_mask=fb, random_seed=seed,
                    length_normalize=length_normalize, tau=tau,
                    stickiness=stick, use_stopping=spec["use_stopping"],
                    fixed_depth=fixed_depth)
                fits[name] = ("hier", paths)

        for test_level in range(2, n_levels + 1):
            true_factor = np.asarray(meta['true_factors'][f'level_{test_level}'], int)
            if not np.any(true_factor == 0):
                continue
            for name in variants:
                fam, obj = fits[name]
                if fam == "flat":
                    m, lvl = probe_flat_exact(obj, true_factor)
                else:
                    m, lvl = probe_hier_exact(obj, true_factor, test_level)
                if m is None:
                    row = {"transfer_acc": np.nan, "transfer_f1": np.nan,
                           "precision": np.nan, "recall": np.nan, "level_used": -1}
                else:
                    row = {"transfer_acc": m["accuracy"], "transfer_f1": m["f1"],
                           "precision": m["precision"], "recall": m["recall"],
                           "level_used": lvl}
                row.update({"alpha": alpha, "omega": omega, "seed": seed,
                            "n_levels": n_levels, "test_level": test_level,
                            "variant": name, "family": fam})
                rows.append(row)

    cols = ["alpha", "omega", "seed", "n_levels", "test_level", "variant", "family",
            "transfer_acc", "transfer_f1", "precision", "recall", "level_used"]
    return pd.DataFrame(rows)[cols]


def run_one_seed(seed, max_levels, cfg: RunConfig):
    """
    Fit each variant once per task and score it with the EXACT transfer_task.py
    readout (single anchor, fixed depth-matched level for hier, scored on all
    trials). Every variant -- flat, sticky-flat, and the four HOLMES lesions --
    goes through the identical metric, so they sit on the same footing as the
    paper's Flat/Hier numbers.

    Reproduction note: for holmes_full (use_stopping=True, stickiness=omega) the
    'transfer_acc' column reproduces the main-text Hier number, and 'flat' the
    main-text Flat number, up to seed set.
    """
    hier_stick = cfg.omega if cfg.hier_stickiness is None else cfg.hier_stickiness
    flat_stick = hier_stick if cfg.flat_stickiness is None else cfg.flat_stickiness
    rows = []

    for n_levels in range(2, max_levels + 1):
        F, meta, fb = generate_scalable_hierarchical_task_quiet(
            n_levels=n_levels, trials_per_context=cfg.trials_per_context, seed=seed)
        n_trials = F.shape[1]; oi = meta['outcome_idx']

        # fit each variant once; reuse across the factors we probe
        fits = {}
        for name in VARIANT_ORDER:
            spec = VARIANTS[name]
            if spec["family"] == "flat":
                stick = flat_stick if spec["flat_stick"] == "on" else 0.0
                # match transfer_task.py: np.random.seed(seed) before flat fit
                np.random.seed(seed)
                particles, _ = one_layer_inference_loop(
                    nTrials=n_trials, nParticles=cfg.n_particles, nFeatures=F.shape[0],
                    regime=cfg.omega, alpha=cfg.alpha, f=F,
                    outcome_idx_per_trial=np.full(n_trials, oi, int),
                    feedback_mask=fb[oi, :], random_seed=seed,
                    length_normalize=cfg.length_normalize, tau=cfg.tau, stickiness=stick)
                fits[name] = ("flat", particles)
            else:
                stick = hier_stick if spec["stick"] == "on" else 0.0
                np.random.seed(seed)
                particles, _, paths = full_hier_inference_loop(
                    nTrials=n_trials, nParticles=cfg.n_particles, nFeatures=F.shape[0],
                    alpha=cfg.alpha, omega=cfg.omega, f=F,
                    max_depth=cfg.max_depth, max_children=cfg.max_children,
                    outcome_idx=oi, feedback_mask=fb, random_seed=seed,
                    length_normalize=cfg.length_normalize, tau=cfg.tau,
                    stickiness=stick, use_stopping=spec["use_stopping"],
                    fixed_depth=cfg.fixed_depth)
                fits[name] = ("hier", paths)

        # probe every latent factor from species (level 2) up
        for test_level in range(2, n_levels + 1):
            true_factor = np.asarray(meta['true_factors'][f'level_{test_level}'], int)
            if not np.any(true_factor == 0):
                continue

            for name in VARIANT_ORDER:
                fam, obj = fits[name]
                if fam == "flat":
                    m, lvl = probe_flat_exact(obj, true_factor)
                else:
                    m, lvl = probe_hier_exact(obj, true_factor, test_level)

                if m is None:
                    row = {"transfer_acc": np.nan, "transfer_f1": np.nan,
                           "precision": np.nan, "recall": np.nan, "level_used": -1}
                else:
                    row = {"transfer_acc": m["accuracy"], "transfer_f1": m["f1"],
                           "precision": m["precision"], "recall": m["recall"],
                           "level_used": lvl}
                row.update({"seed": seed, "n_levels": n_levels, "test_level": test_level,
                            "variant": name, "family": fam})
                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# REPORTING
# ============================================================

def _paired(df, a, b, metric):
    pa = df[df.variant == a].groupby("seed")[metric].mean()
    pb = df[df.variant == b].groupby("seed")[metric].mean()
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common] - pb.loc[common]).dropna()
    mean = float(d.mean()) if len(d) else np.nan
    sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    t = p = np.nan
    if len(d) > 1 and sd and sd > 0:
        try:
            from scipy import stats as st
            t, p = st.ttest_rel(pa.loc[common], pb.loc[common]); t, p = float(t), float(p)
        except Exception:
            pass
    return mean, sd, t, p, len(d)


def report(df):
    print("\n" + "=" * 96)
    print("PER-VARIANT MEANS (exact transfer_task.py readout; averaged over seeds x factors)")
    print("=" * 96)
    agg = (df.groupby("variant")
             .agg(transfer_acc=("transfer_acc", "mean"),
                  transfer_f1=("transfer_f1", "mean"),
                  precision=("precision", "mean"),
                  recall=("recall", "mean"),
                  level_used=("level_used", "mean"))
             .reindex(VARIANT_ORDER))
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(agg.to_string())

    print("\n" + "=" * 96)
    print("TRANSFER ACCURACY BY FACTOR (test_level) — the main-text view")
    print("=" * 96)
    piv = (df.pivot_table(index="variant", columns="test_level",
                          values="transfer_acc", aggfunc="mean")
             .reindex(VARIANT_ORDER))
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(piv.to_string())
    print("  (For holmes_full this row reproduces the paper's Hier numbers; 'flat' the")
    print("   paper's Flat numbers. Deeper test_level = more abstract factor.)")

    print("\n" + "=" * 96)
    print("KEY CONTRASTS (paired across seeds on per-seed mean transfer_acc)")
    print("=" * 96)

    def line(label, a, b, metric="transfer_acc"):
        m, sd, t, p, n = _paired(df, a, b, metric)
        sd_s = f"{sd:.3f}" if np.isfinite(sd) else " nan"
        t_s = f"{t:+.2f}" if np.isfinite(t) else " nan"
        p_s = f"{p:.4f}" if np.isfinite(p) else " nan"
        return f"  {label:<30} {a}-{b}: {m:+.3f} +/- {sd_s} (t={t_s}, p={p_s}, n={n})"

    print("\n[Hierarchy vs flat]")
    print(line("HOLMES-full vs flat", "holmes_full", "flat"))
    print(line("HOLMES-full vs sticky-flat", "holmes_full", "flat_sticky"))
    print("\n[Is the combination needed vs just structure?]")
    print(line("full vs structure-only", "holmes_full", "holmes_struct_only"))
    print(line("value of stopping", "holmes_full", "holmes_no_stop"))
    print(line("value of stickiness", "holmes_full", "holmes_no_stick"))
    print("\n[Flat enhancement]")
    print(line("sticky-flat vs flat", "flat_sticky", "flat"))

    print("\n" + "=" * 96)
    print("DEEP-LEVEL CONTRASTS (test_level >= 3 only — where hierarchy is expected to matter)")
    print("=" * 96)
    deep = df[df.test_level >= 3]
    if len(deep):
        for lbl, a, b in [("HOLMES-full vs flat", "holmes_full", "flat"),
                          ("full vs structure-only", "holmes_full", "holmes_struct_only"),
                          ("value of stopping", "holmes_full", "holmes_no_stop"),
                          ("value of stickiness", "holmes_full", "holmes_no_stick")]:
            m, sd, t, p, n = _paired(deep, a, b, "transfer_acc")
            sd_s = f"{sd:.3f}" if np.isfinite(sd) else " nan"
            t_s = f"{t:+.2f}" if np.isfinite(t) else " nan"
            p_s = f"{p:.4f}" if np.isfinite(p) else " nan"
            print(f"  {lbl:<30} {a}-{b}: {m:+.3f} +/- {sd_s} (t={t_s}, p={p_s}, n={n})")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Efficient hierarchical transfer probe (option C)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--max_levels", type=int, default=5)
    ap.add_argument("--n_particles", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--max_depth", type=int, default=20)
    ap.add_argument("--max_children", type=int, default=20)
    ap.add_argument("--fixed_depth", type=int, default=6,
                    help="Descent depth for the no-stopping lesion (decoupled from max_depth).")
    ap.add_argument("--length_normalize", action="store_true")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--hier_stickiness", type=float, default=None)
    ap.add_argument("--flat_stickiness", type=float, default=None)
    ap.add_argument("--trials_per_context", type=int, default=10)
    ap.add_argument("--out_prefix", type=str, default="transfer_probe")
    args = ap.parse_args()

    cfg = RunConfig(
        n_particles=args.n_particles, alpha=args.alpha, omega=args.omega,
        max_depth=args.max_depth, max_children=args.max_children,
        fixed_depth=args.fixed_depth,
        length_normalize=args.length_normalize, tau=args.tau,
        hier_stickiness=args.hier_stickiness, flat_stickiness=args.flat_stickiness,
        trials_per_context=args.trials_per_context,
    )

    print("=" * 96)
    print("HIERARCHICAL TRANSFER — ablation on the EXACT main-text readout")
    print("=" * 96)
    print(f"Variants: {VARIANT_ORDER}")
    print(f"Seeds: {args.seeds} | max_levels={args.max_levels} | particles={args.n_particles}")
    print(f"alpha={args.alpha} omega={args.omega} depth={args.max_depth} children={args.max_children}")
    print(f"length_normalize={args.length_normalize}")

    import time
    dfs = []
    for seed in args.seeds:
        t0 = time.time()
        d = run_one_seed(seed, args.max_levels, cfg)
        dfs.append(d)
        ho = d[d.variant == "holmes_full"]["transfer_acc"].mean()
        fl = d[d.variant == "flat"]["transfer_acc"].mean()
        print(f"  seed {seed}: {time.time()-t0:.1f}s | transfer_acc holmes_full={ho:.3f} flat={fl:.3f}")

    df = pd.concat(dfs, ignore_index=True)
    report(df)

    import datetime as dt
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = f"{args.out_prefix}_{ts}_long.csv"; df.to_csv(csv, index=False)
    summ = f"{args.out_prefix}_{ts}_summary.csv"
    df.groupby(["variant"]).mean(numeric_only=True).reindex(VARIANT_ORDER).to_csv(summ)
    with open(f"{args.out_prefix}_{ts}_meta.json", "w") as fh:
        json.dump({"args": vars(args), "variants": VARIANTS}, fh, indent=2)
    print(f"Saved: {csv}\nSaved: {summ}")


if __name__ == "__main__":
    main()