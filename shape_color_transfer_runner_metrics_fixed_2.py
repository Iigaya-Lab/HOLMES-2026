#!/usr/bin/env python3
"""
Shape x many-colors transfer runner for HOLMES vs flat LC model.

Task:
- 2 shapes, many colors.
- Reward depends on shape, color irrelevant.
- Train on a subset of colors; test on held-out colors.
- Critical metric: first prediction for each held-out color before it has prior feedback.

Fairness:
- Same observations, same online reward target.
- No oracle decoder, no post-hoc reward rates.
- HOLMES has no prescribed number of levels; max_depth is only computational.
"""
from __future__ import annotations

import argparse, importlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from flat_model import one_layer_inference_loop
from hier_model import full_hier_inference_loop


@dataclass
class ShapeColorConfig:
    n_colors: int = 10
    n_train_colors: int = 8
    n_test_colors: int = 2
    train_reps_per_stimulus: int = 10
    test_reps_per_stimulus: int = 1
    n_shape_copies: int = 2
    n_color_copies: int = 4
    n_nuisance_features: int = 0
    blocked_train_by_shape: bool = True
    shuffle_within_train_blocks: bool = True
    shuffle_test: bool = True
    reward_rule: str = "shape0"  # or random_balanced


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
        raise ValueError("reward_rule must be 'shape0' or 'random_balanced'")

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
        if rng.random() < 0.5:
            train_trials = blocks[0] + blocks[1]
        else:
            train_trials = blocks[1] + blocks[0]
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
        "n_trials": len(trials), "n_train_trials": len(train_trials), "n_test_trials": len(test_trials),
        "outcome_idx": outcome_idx, "n_nonoutcome": n_nonoutcome,
        "n_shape_feats": n_shape_feats, "n_color_feats": n_color_feats,
        "rewarded_shape": rewarded_shape, "train_colors": train_colors, "test_colors": test_colors,
        "trials": [{k: v for k, v in tr.items() if k != "features"} for tr in trials],
    }
    return F, meta, feedback_mask


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


def acc(prob: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    return float(np.mean((prob[mask] > 0.5).astype(int) == y[mask]))



def brier_score(prob: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared probabilistic prediction error; lower is better."""
    if not np.any(mask):
        return np.nan
    p = np.asarray(prob[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    return float(np.mean((p - yy) ** 2))


def log_loss_score(prob: np.ndarray, y: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> float:
    """Binary log loss; lower is better."""
    if not np.any(mask):
        return np.nan
    p = np.clip(np.asarray(prob[mask], dtype=float), eps, 1.0 - eps)
    yy = np.asarray(y[mask], dtype=float)
    return float(np.mean(-(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p))))


def shape_probability_summary(prob: np.ndarray, shape: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean predicted P(reward=1) by shape and shape contrast."""
    out = {}
    for s in [0, 1]:
        smask = mask & (shape == s)
        out[f"shape{s}_mean_prob"] = float(np.mean(prob[smask])) if np.any(smask) else np.nan
    out["shape_contrast"] = out["shape0_mean_prob"] - out["shape1_mean_prob"]
    return out

def pairwise_same_group_same_node(assignments: np.ndarray, group: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    idx = np.where(mask & (assignments >= 0))[0]
    if len(idx) < 2:
        return {"same_group_same_node": np.nan, "diff_group_same_node": np.nan, "separation": np.nan}
    same_total = same_hit = diff_total = diff_hit = 0
    max_pairs = 30000
    pairs = 0
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
    return {"same_group_same_node": same, "diff_group_same_node": diff,
            "separation": same - diff if np.isfinite(same) and np.isfinite(diff) else np.nan}


def choose_best_hier_level_for_shape(paths: np.ndarray, shapes: np.ndarray, train_mask: np.ndarray):
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


def load_hier_loop(module_name: str):
    return importlib.import_module(module_name).full_hier_inference_loop


def run_one_seed(seed: int, cfg: ShapeColorConfig, hier_module: str, n_particles: int,
                 alpha: float, omega: float, max_depth: int, max_children: int,
                 length_normalize: bool, tau: float):
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

    p_flat, _, r_flat, _ = one_layer_inference_loop(
        nTrials=F.shape[1], nParticles=n_particles, nFeatures=F.shape[0],
        regime=omega, alpha=alpha, f=F,
        outcome_idx_per_trial=np.full(F.shape[1], oi, dtype=int),
        feedback_mask=fb[oi, :], random_seed=seed,
        length_normalize=length_normalize, tau=tau,
    )

    p_hier, _, r_hier, _, _, paths_hier, _ = full_hier_inference_loop(
        nTrials=F.shape[1], nParticles=n_particles, nFeatures=F.shape[0],
        alpha=alpha, omega=omega, f=F, max_depth=max_depth, max_children=max_children,
        outcome_idx=oi, feedback_mask=fb, random_seed=seed,
        length_normalize=length_normalize,tau=tau,
    )

    rewarded_shape = int(meta["rewarded_shape"])
    # Sign for correcting contrast when reward_rule=random_balanced:
    # positive = model assigns higher P(reward) to the actual rewarded shape.
    sign = 1 if rewarded_shape == 0 else -1

    flat_assign = get_flat_assignments(p_flat)
    flat_shape_diag = pairwise_same_group_same_node(flat_assign, shape, train_mask)
    flat_color_diag = pairwise_same_group_same_node(flat_assign, color, train_mask)
    best_L, hier_info = choose_best_hier_level_for_shape(paths_hier, shape, train_mask)
    hier_best_assign = get_hier_level_assignments(paths_hier, best_L) if best_L >= 0 else np.full(F.shape[1], -1)
    hier_color_diag = pairwise_same_group_same_node(hier_best_assign, color, train_mask)

    flat_train_shape = shape_probability_summary(r_flat, shape, train_mask)
    hier_train_shape = shape_probability_summary(r_hier, shape, train_mask)
    flat_held_shape  = shape_probability_summary(r_flat, shape, first_heldout_mask)
    hier_held_shape  = shape_probability_summary(r_hier, shape, first_heldout_mask)

    def acc_shape(prob, y_, mask_, s):
        m = mask_ & (shape == s)
        return acc(prob, y_, m)

    result = {
        # ── identity ──────────────────────────────────────────────
        "seed":           seed,
        "rewarded_shape": rewarded_shape,
        "n_train":        int(train_mask.sum()),
        "n_first_heldout": int(first_heldout_mask.sum()),

        # ── train accuracy ────────────────────────────────────────
        "flat_train_acc": acc(r_flat, y, train_mask),
        "hier_train_acc": acc(r_hier, y, train_mask),
        "adv_train_acc":  acc(r_hier, y, train_mask) - acc(r_flat, y, train_mask),

        # ── first-heldout accuracy ────────────────────────────────
        "flat_first_heldout_acc": acc(r_flat, y, first_heldout_mask),
        "hier_first_heldout_acc": acc(r_hier, y, first_heldout_mask),
        "adv_first_heldout_acc":  acc(r_hier, y, first_heldout_mask) - acc(r_flat, y, first_heldout_mask),

        # ── per-shape heldout accuracy ────────────────────────────
        "flat_heldout_shape0_acc": acc_shape(r_flat, y, first_heldout_mask, 0),
        "flat_heldout_shape1_acc": acc_shape(r_flat, y, first_heldout_mask, 1),
        "hier_heldout_shape0_acc": acc_shape(r_hier, y, first_heldout_mask, 0),
        "hier_heldout_shape1_acc": acc_shape(r_hier, y, first_heldout_mask, 1),
        "adv_heldout_shape0_acc":  acc_shape(r_hier, y, first_heldout_mask, 0) - acc_shape(r_flat, y, first_heldout_mask, 0),
        "adv_heldout_shape1_acc":  acc_shape(r_hier, y, first_heldout_mask, 1) - acc_shape(r_flat, y, first_heldout_mask, 1),

        # ── probabilistic metrics ─────────────────────────────────
        "flat_train_brier":          brier_score(r_flat, y, train_mask),
        "hier_train_brier":          brier_score(r_hier, y, train_mask),
        "adv_train_brier_reduction": brier_score(r_flat, y, train_mask) - brier_score(r_hier, y, train_mask),
        "flat_heldout_brier":          brier_score(r_flat, y, first_heldout_mask),
        "hier_heldout_brier":          brier_score(r_hier, y, first_heldout_mask),
        "adv_heldout_brier_reduction": brier_score(r_flat, y, first_heldout_mask) - brier_score(r_hier, y, first_heldout_mask),

        "flat_train_logloss":          log_loss_score(r_flat, y, train_mask),
        "hier_train_logloss":          log_loss_score(r_hier, y, train_mask),
        "adv_train_logloss_reduction": log_loss_score(r_flat, y, train_mask) - log_loss_score(r_hier, y, train_mask),
        "flat_heldout_logloss":          log_loss_score(r_flat, y, first_heldout_mask),
        "hier_heldout_logloss":          log_loss_score(r_hier, y, first_heldout_mask),
        "adv_heldout_logloss_reduction": log_loss_score(r_flat, y, first_heldout_mask) - log_loss_score(r_hier, y, first_heldout_mask),

        # ── unsigned shape contrast ───────────────────────────────
        "flat_train_shape0_mean_prob": flat_train_shape["shape0_mean_prob"],
        "flat_train_shape1_mean_prob": flat_train_shape["shape1_mean_prob"],
        "flat_train_shape_contrast":   flat_train_shape["shape_contrast"],
        "hier_train_shape0_mean_prob": hier_train_shape["shape0_mean_prob"],
        "hier_train_shape1_mean_prob": hier_train_shape["shape1_mean_prob"],
        "hier_train_shape_contrast":   hier_train_shape["shape_contrast"],
        "adv_train_shape_contrast":    hier_train_shape["shape_contrast"] - flat_train_shape["shape_contrast"],

        "flat_heldout_shape0_mean_prob": flat_held_shape["shape0_mean_prob"],
        "flat_heldout_shape1_mean_prob": flat_held_shape["shape1_mean_prob"],
        "flat_heldout_shape_contrast":   flat_held_shape["shape_contrast"],
        "hier_heldout_shape0_mean_prob": hier_held_shape["shape0_mean_prob"],
        "hier_heldout_shape1_mean_prob": hier_held_shape["shape1_mean_prob"],
        "hier_heldout_shape_contrast":   hier_held_shape["shape_contrast"],
        "adv_heldout_shape_contrast":    hier_held_shape["shape_contrast"] - flat_held_shape["shape_contrast"],

        # ── signed contrast (directly averageable across random_balanced seeds) ──
        "flat_train_signed_contrast":   sign * flat_train_shape["shape_contrast"],
        "hier_train_signed_contrast":   sign * hier_train_shape["shape_contrast"],
        "adv_train_signed_contrast":    sign * (hier_train_shape["shape_contrast"] - flat_train_shape["shape_contrast"]),
        "flat_heldout_signed_contrast": sign * flat_held_shape["shape_contrast"],
        "hier_heldout_signed_contrast": sign * hier_held_shape["shape_contrast"],
        "adv_heldout_signed_contrast":  sign * (hier_held_shape["shape_contrast"] - flat_held_shape["shape_contrast"]),

        # ── representational structure ────────────────────────────
        "flat_n_clusters": int(len(np.unique(flat_assign[flat_assign >= 0]))),
        "hier_nodes_by_level": count_hier_nodes(paths_hier),
        "hier_best_shape_level": int(best_L),
        "hier_best_shape_separation": float(hier_info["best_diag"].get("separation", np.nan)),
        "flat_shape_same_node": flat_shape_diag["same_group_same_node"],
        "flat_shape_diff_node": flat_shape_diag["diff_group_same_node"],
        "flat_shape_separation": flat_shape_diag["separation"],
        "hier_shape_same_node": hier_info["best_diag"].get("same_group_same_node", np.nan),
        "hier_shape_diff_node": hier_info["best_diag"].get("diff_group_same_node", np.nan),
        "flat_color_separation": flat_color_diag["separation"],
        "hier_color_separation_at_shape_level": hier_color_diag["separation"],
        "hier_diag_by_level": hier_info["diag_by_level"],
    }

    # ── full per-trial DataFrame (every trial, both models) ──────
    all_rows = []
    for t in range(len(trials)):
        flat_p = float(r_flat[t])
        hier_p = float(r_hier[t])
        reward = int(y[t])
        sh     = int(shape[t])
        correct_dir = 1 if sh == rewarded_shape else -1
        all_rows.append({
            "seed":              seed,
            "rewarded_shape":    rewarded_shape,
            "trial":             t,
            "phase":             phase[t],
            "shape":             sh,
            "color":             int(color[t]),
            "is_first_heldout":  bool(first_heldout_mask[t]),
            "reward":            reward,
            "flat_prob":         flat_p,
            "hier_prob":         hier_p,
            "flat_signed_margin":  correct_dir * (flat_p - 0.5),
            "hier_signed_margin":  correct_dir * (hier_p - 0.5),
            "flat_correct":      int((flat_p > 0.5) == reward),
            "hier_correct":      int((hier_p > 0.5) == reward),
            "flat_cluster":      int(flat_assign[t]),
            "hier_best_shape_level": int(best_L),
            "hier_best_node":    int(hier_best_assign[t]),
        })
    all_trials_df = pd.DataFrame(all_rows)

    # ── first-heldout subset (kept separately for quick access) ──
    rows = []
    for t in np.where(first_heldout_mask)[0]:
        rows.append({k: all_rows[t][k] for k in all_rows[t]})
    return result, pd.DataFrame(rows), all_trials_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hier_module", type=str, default="full_hier_parent_specific_ncrp_confident_readout")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_particles", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--max_children", type=int, default=20)
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed per-seed diagnostics: mean probabilities by shape, all-level hierarchy diagnostics, and held-out trials.")
    parser.add_argument("--no_length_normalize", action="store_true")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--n_colors", type=int, default=10)
    parser.add_argument("--n_train_colors", type=int, default=8)
    parser.add_argument("--n_test_colors", type=int, default=2)
    parser.add_argument("--train_reps", type=int, default=5)
    parser.add_argument("--test_reps", type=int, default=1)
    parser.add_argument("--shape_copies", type=int, default=6)
    parser.add_argument("--color_copies", type=int, default=1)
    parser.add_argument("--nuisance_features", type=int, default=0)
    parser.add_argument("--shuffled_train", action="store_true")
    parser.add_argument("--reward_rule", type=str, default="shape0", choices=["shape0", "random_balanced"])
    parser.add_argument("--out_prefix", type=str, default="/mnt/data/shape_color_transfer")
    args = parser.parse_args()

    cfg = ShapeColorConfig(
        n_colors=args.n_colors, n_train_colors=args.n_train_colors, n_test_colors=args.n_test_colors,
        train_reps_per_stimulus=args.train_reps, test_reps_per_stimulus=args.test_reps,
        n_shape_copies=args.shape_copies, n_color_copies=args.color_copies,
        n_nuisance_features=args.nuisance_features, blocked_train_by_shape=not args.shuffled_train,
        reward_rule=args.reward_rule,
    )

    print("=" * 88)
    print("SHAPE x MANY COLORS TRANSFER TASK")
    print("=" * 88)
    print(f"Hier module: {args.hier_module}")
    print(f"Seeds: {args.seeds}")
    print(f"alpha={args.alpha}, omega={args.omega}, particles={args.n_particles}")
    print(f"length_normalize={not args.no_length_normalize}, tau={args.tau}")
    print(f"Task config: {cfg}")

    results, trial_dfs, all_trial_dfs = [], [], []
    for seed in args.seeds:
        print(f"\nRunning seed {seed}...")
        res, trial_df, all_trials_df = run_one_seed(
            seed, cfg, args.hier_module, args.n_particles, args.alpha, args.omega,
            args.max_depth, args.max_children, 
            not args.no_length_normalize, args.tau, 
        )
        results.append(res); trial_dfs.append(trial_df); all_trial_dfs.append(all_trials_df)
        print(f"  first-heldout F/H={res['flat_first_heldout_acc']:.3f}/{res['hier_first_heldout_acc']:.3f} adv={res['adv_first_heldout_acc']:+.3f}; train F/H={res['flat_train_acc']:.3f}/{res['hier_train_acc']:.3f}")
        print(f"  shape structure: flat sep={res['flat_shape_separation']:.3f}; hier L{res['hier_best_shape_level']} sep={res['hier_best_shape_separation']:.3f}; nodes={res['hier_nodes_by_level']}")
        if args.debug:
            print("  DEBUG mean predicted P(reward=1) by shape:")
            print(f"    train flat: shape0={res['flat_train_shape0_mean_prob']:.3f}, shape1={res['flat_train_shape1_mean_prob']:.3f}, contrast={res['flat_train_shape_contrast']:+.3f}")
            print(f"    train hier: shape0={res['hier_train_shape0_mean_prob']:.3f}, shape1={res['hier_train_shape1_mean_prob']:.3f}, contrast={res['hier_train_shape_contrast']:+.3f}")
            print(f"    heldout flat: shape0={res['flat_heldout_shape0_mean_prob']:.3f}, shape1={res['flat_heldout_shape1_mean_prob']:.3f}, contrast={res['flat_heldout_shape_contrast']:+.3f}")
            print(f"    heldout hier: shape0={res['hier_heldout_shape0_mean_prob']:.3f}, shape1={res['hier_heldout_shape1_mean_prob']:.3f}, contrast={res['hier_heldout_shape_contrast']:+.3f}")
            print("  DEBUG probabilistic scores:")
            print(f"    train Brier F/H={res['flat_train_brier']:.4f}/{res['hier_train_brier']:.4f}; reduction={res['adv_train_brier_reduction']:+.4f}")
            print(f"    heldout Brier F/H={res['flat_heldout_brier']:.4f}/{res['hier_heldout_brier']:.4f}; reduction={res['adv_heldout_brier_reduction']:+.4f}")
            print(f"    train logloss F/H={res['flat_train_logloss']:.4f}/{res['hier_train_logloss']:.4f}; reduction={res['adv_train_logloss_reduction']:+.4f}")
            print(f"    heldout logloss F/H={res['flat_heldout_logloss']:.4f}/{res['hier_heldout_logloss']:.4f}; reduction={res['adv_heldout_logloss_reduction']:+.4f}")
            print("  DEBUG hierarchy shape diagnostics by level:")
            for level_name, diag in res.get('hier_diag_by_level', {}).items():
                print(f"    {level_name}: same={diag['same_group_same_node']:.3f}, diff={diag['diff_group_same_node']:.3f}, sep={diag['separation']:.3f}")
            print("  DEBUG first-heldout trial predictions:")
            if len(trial_df):
                cols_dbg = ['trial', 'shape', 'color', 'reward', 'flat_prob', 'hier_prob',
                            'flat_signed_margin', 'hier_signed_margin',
                            'flat_correct', 'hier_correct', 'flat_cluster', 'hier_best_node']
                cols_dbg = [c for c in cols_dbg if c in trial_df.columns]
                print(trial_df[cols_dbg].to_string(index=False))

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "hier_diag_by_level"} for r in results])
    trials_df     = pd.concat(trial_dfs,     ignore_index=True) if trial_dfs     else pd.DataFrame()
    all_trials_df = pd.concat(all_trial_dfs, ignore_index=True) if all_trial_dfs else pd.DataFrame()

    print("\n" + "=" * 88)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 88)
    cols = [
        "seed",
        "flat_train_acc", "hier_train_acc",
        "flat_first_heldout_acc", "hier_first_heldout_acc", "adv_first_heldout_acc",
        "flat_heldout_shape_contrast", "hier_heldout_shape_contrast", "adv_heldout_shape_contrast",
        "flat_heldout_brier", "hier_heldout_brier", "adv_heldout_brier_reduction",
        "flat_heldout_logloss", "hier_heldout_logloss", "adv_heldout_logloss_reduction",
        "flat_shape_separation", "hier_best_shape_level", "hier_best_shape_separation",
        "flat_n_clusters", "hier_nodes_by_level"
    ]
    print(df[cols].to_string(index=False))
    print("\nMean first-heldout accuracy:")
    print(f"  Flat: {df['flat_first_heldout_acc'].mean():.3f} +/- {df['flat_first_heldout_acc'].std(ddof=1):.3f}")
    print(f"  Hier: {df['hier_first_heldout_acc'].mean():.3f} +/- {df['hier_first_heldout_acc'].std(ddof=1):.3f}")
    print(f"  Adv : {df['adv_first_heldout_acc'].mean():+.3f} +/- {df['adv_first_heldout_acc'].std(ddof=1):.3f}")
    adv = df['adv_first_heldout_acc']
    from scipy import stats as _stats
    t_stat, p_val = _stats.ttest_1samp(adv.dropna(), 0)
    d = adv.mean() / adv.std(ddof=1)
    print(f"  t({len(adv)-1}) = {t_stat:+.3f}, p = {p_val:.4f}, d = {d:.3f}")

    print("\nMean heldout signed shape contrast (sign-corrected; higher = model reads rewarded shape):")
    print(f"  Flat: {df['flat_heldout_signed_contrast'].mean():+.4f} +/- {df['flat_heldout_signed_contrast'].std(ddof=1):.4f}")
    print(f"  Hier: {df['hier_heldout_signed_contrast'].mean():+.4f} +/- {df['hier_heldout_signed_contrast'].std(ddof=1):.4f}")
    print(f"  Adv : {df['adv_heldout_signed_contrast'].mean():+.4f} +/- {df['adv_heldout_signed_contrast'].std(ddof=1):.4f}")
    adv_sc = df['adv_heldout_signed_contrast']
    t_sc, p_sc = _stats.ttest_1samp(adv_sc.dropna(), 0)
    d_sc = adv_sc.mean() / adv_sc.std(ddof=1)
    print(f"  t({len(adv_sc)-1}) = {t_sc:+.3f}, p = {p_sc:.4f}, d = {d_sc:.3f}")

    print("\nMean heldout shape contrast (unsigned, shape0 - shape1):")
    print(f"  Flat: {df['flat_heldout_shape_contrast'].mean():+.4f} +/- {df['flat_heldout_shape_contrast'].std(ddof=1):.4f}")
    print(f"  Hier: {df['hier_heldout_shape_contrast'].mean():+.4f} +/- {df['hier_heldout_shape_contrast'].std(ddof=1):.4f}")
    print(f"  Adv : {df['adv_heldout_shape_contrast'].mean():+.4f} +/- {df['adv_heldout_shape_contrast'].std(ddof=1):.4f}")

    print("\nMean heldout Brier score (lower is better):")
    print(f"  Flat: {df['flat_heldout_brier'].mean():.4f} +/- {df['flat_heldout_brier'].std(ddof=1):.4f}")
    print(f"  Hier: {df['hier_heldout_brier'].mean():.4f} +/- {df['hier_heldout_brier'].std(ddof=1):.4f}")
    print(f"  Reduction (Flat-Hier): {df['adv_heldout_brier_reduction'].mean():+.4f} +/- {df['adv_heldout_brier_reduction'].std(ddof=1):.4f}")
    adv_b = df['adv_heldout_brier_reduction']
    t_b, p_b = _stats.ttest_1samp(adv_b.dropna(), 0)
    d_b = adv_b.mean() / adv_b.std(ddof=1)
    print(f"  t({len(adv_b)-1}) = {t_b:+.3f}, p = {p_b:.4f}, d = {d_b:.3f}")

    print("\nMean heldout log loss (lower is better):")
    print(f"  Flat: {df['flat_heldout_logloss'].mean():.4f} +/- {df['flat_heldout_logloss'].std(ddof=1):.4f}")
    print(f"  Hier: {df['hier_heldout_logloss'].mean():.4f} +/- {df['hier_heldout_logloss'].std(ddof=1):.4f}")
    print(f"  Reduction (Flat-Hier): {df['adv_heldout_logloss_reduction'].mean():+.4f} +/- {df['adv_heldout_logloss_reduction'].std(ddof=1):.4f}")
    adv_l = df['adv_heldout_logloss_reduction']
    t_l, p_l = _stats.ttest_1samp(adv_l.dropna(), 0)
    d_l = adv_l.mean() / adv_l.std(ddof=1)
    print(f"  t({len(adv_l)-1}) = {t_l:+.3f}, p = {p_l:.4f}, d = {d_l:.3f}")


    import datetime as _dt
    run_timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_prefix      = Path(args.out_prefix)
    csv_path        = f"{out_prefix.name}_{run_timestamp}.csv"
    trials_path     = f"{out_prefix.name}_{run_timestamp}_first_heldout_trials.csv"
    all_trials_path = f"{out_prefix.name}_{run_timestamp}_all_trials.csv"
    json_path       = f"{out_prefix.name}_{run_timestamp}.json"

    df.to_csv(csv_path, index=False)
    trials_df.to_csv(trials_path, index=False)
    all_trials_df.to_csv(all_trials_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"config": asdict(cfg), "args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved summary CSV       → {csv_path}")
    print(f"Saved first-heldout CSV → {trials_path}")
    print(f"Saved all-trials CSV    → {all_trials_path}")
    print(f"Saved full JSON         → {json_path}")


if __name__ == "__main__":
    main()