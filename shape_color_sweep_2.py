#!/usr/bin/env python3
"""
shape_color_sweep.py

Parameter sweep for the Shape × Many Colors transfer task.

Follows the same structure as transfer_task.py:
    run_single_seed  ->  run_parallel_analysis  ->  run_parameter_sweep

Primary metrics (all continuous, threshold-independent):
    adv_heldout_brier_reduction   : Flat - Hier Brier score, higher is better
    adv_heldout_logloss_reduction : Flat - Hier log loss, higher is better
    adv_heldout_shape_contrast    : Hier - Flat shape contrast, higher is better

Secondary metric:
    adv_first_heldout_acc         : Hier - Flat accuracy on first heldout trials

The key ablation — tree_mix=0.0 (leaf-only) vs tree_mix=1.0 (path readout) —
is built into the sweep via --tree_mix_vals.  If tree_mix=1.0 consistently
outperforms tree_mix=0.0, the path readout is confirmed as the mechanism.

Usage
-----
    # Pure alpha × omega sweep (default — all other params fixed)
    python shape_color_sweep.py --n_param_settings 50 --n_seeds 50 \
        --shuffled_train --save --out_prefix results/shape_color_sweep

    # Add tree_mix ablation on top of the alpha × omega sweep
    python shape_color_sweep.py --n_param_settings 50 --n_seeds 50 \
        --tree_mix_vals 0.0 1.0 \
        --shuffled_train --save --out_prefix results/shape_color_ablation

    # Full grid including depth_scale and stick_scale
    python shape_color_sweep.py --n_param_settings 50 --n_seeds 50 \
        --tree_mix_vals 0.0 1.0 \
        --depth_scale_vals 0.25 0.5 1.0 \
        --stick_scale_vals 0.25 0.5 1.0 \
        --shuffled_train --save --out_prefix results/shape_color_full
"""

import argparse
import datetime
import itertools
import pickle
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from shape_color_transfer_runner_metrics_fixed_2 import (
    ShapeColorConfig,
    generate_shape_color_transfer_task,
    run_one_seed as _runner_run_one_seed,
)
from flat_model import one_layer_inference_loop
from hier_model import full_hier_inference_loop


# ============================================================================
# UTILITY
# ============================================================================

def log_progress(message, flush=True):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=flush)


# ============================================================================
# SINGLE-SEED WORKER
# ============================================================================

def run_single_seed(
    seed,
    # task config
    n_colors,
    n_train_colors,
    n_test_colors,
    train_reps,
    test_reps,
    shape_copies,
    color_copies,
    n_nuisance_features,
    blocked_train_by_shape,
    reward_rule,
    # model config
    n_particles,
    alpha,
    omega,
    max_depth,
    max_children,
    #level_decay,
    length_normalize,
    tau,
):
    """
    Run flat and hierarchical models on one seed of the shape-color task.

    Returns a single flat dict of every scalar that can be derived from
    this seed run, tagged with the parameter identity columns.  The goal
    is that if you think of a new analysis metric after the sweep you
    can compute it from the saved per-seed CSV without re-running.

    Fields returned
    ---------------
    Parameter identity
        seed, alpha, omega, n_particles, train_reps, test_reps,
        shape_copies, color_copies, blocked_train_by_shape, reward_rule

    Task metadata
        n_train, n_first_heldout, n_colors, n_train_colors, n_test_colors,
        rewarded_shape

    Train accuracy & probabilistic scores  (flat / hier / advantage)
        flat_train_acc, hier_train_acc, adv_train_acc
        flat_train_brier, hier_train_brier, adv_train_brier_reduction
        flat_train_logloss, hier_train_logloss, adv_train_logloss_reduction
        flat_train_shape_contrast, hier_train_shape_contrast, adv_train_shape_contrast
        flat_train_shape0_mean_prob, flat_train_shape1_mean_prob
        hier_train_shape0_mean_prob, hier_train_shape1_mean_prob

    First-heldout accuracy & probabilistic scores  (flat / hier / advantage)
        flat_first_heldout_acc, hier_first_heldout_acc, adv_first_heldout_acc
        flat_heldout_brier, hier_heldout_brier, adv_heldout_brier_reduction
        flat_heldout_logloss, hier_heldout_logloss, adv_heldout_logloss_reduction
        flat_heldout_shape_contrast, hier_heldout_shape_contrast, adv_heldout_shape_contrast
        flat_heldout_shape0_mean_prob, flat_heldout_shape1_mean_prob
        hier_heldout_shape0_mean_prob, hier_heldout_shape1_mean_prob

    All-heldout accuracy & probabilistic scores  (flat / hier / advantage)
        flat_all_heldout_acc, hier_all_heldout_acc, adv_all_heldout_acc
        flat_all_heldout_brier, hier_all_heldout_brier, adv_all_heldout_brier_reduction
        flat_all_heldout_logloss, hier_all_heldout_logloss, adv_all_heldout_logloss_reduction
        flat_all_heldout_shape_contrast, hier_all_heldout_shape_contrast

    Per-shape heldout accuracy  (shape0 / shape1 separately, flat / hier)
        flat_heldout_shape0_acc, flat_heldout_shape1_acc
        hier_heldout_shape0_acc, hier_heldout_shape1_acc
        adv_heldout_shape0_acc, adv_heldout_shape1_acc

    Representational structure
        flat_n_clusters
        flat_shape_separation, flat_shape_same_node, flat_shape_diff_node
        flat_color_separation
        hier_best_shape_level, hier_best_shape_separation
        hier_shape_same_node, hier_shape_diff_node
        hier_color_separation_at_shape_level
        hier_n_levels_used        (deepest level with any nodes)
        hier_total_nodes          (sum of nodes across all levels)
        hier_nodes_level_{L}      (node count at each level L, L=0..max_used)

    Node diagnostics  (post-hoc, not used for prediction)
        hier_top_reward_node_level, hier_top_reward_node, hier_top_reward_node_rate,
        hier_top_reward_node_p_shape0, hier_top_reward_node_n
        hier_best_level_mean_reward_rate, hier_best_level_max_informativeness
    """
    cfg = ShapeColorConfig(
        n_colors=n_colors,
        n_train_colors=n_train_colors,
        n_test_colors=n_test_colors,
        train_reps_per_stimulus=train_reps,
        test_reps_per_stimulus=test_reps,
        n_shape_copies=shape_copies,
        n_color_copies=color_copies,
        n_nuisance_features=n_nuisance_features,
        blocked_train_by_shape=blocked_train_by_shape,
        reward_rule=reward_rule,
    )

    res, trial_df, full_df = _runner_run_one_seed(
        seed=seed,
        cfg=cfg,
        n_particles=n_particles,
        alpha=alpha,
        omega=omega,
        max_depth=max_depth,
        max_children=max_children,
        #level_decay=level_decay,
        length_normalize=length_normalize,
        tau=tau,
    )

    # ── parameter identity ────────────────────────────────────────
    out = {
        'seed':                seed,
        'alpha':               alpha,
        'omega':               omega,
        #'level_decay':         level_decay,
        'n_particles':         n_particles,
        'train_reps':          train_reps,
        'test_reps':           test_reps,
        'shape_copies':        shape_copies,
        'color_copies':        color_copies,
        'blocked_train':       int(blocked_train_by_shape),
        'reward_rule':         reward_rule,
        'n_colors':            n_colors,
        'n_train_colors':      n_train_colors,
        'n_test_colors':       n_test_colors,
    }

    # ── copy every scalar from runner result dict ─────────────────
    # Drop only the non-serialisable nested dict hier_diag_by_level.
    # Everything else — including fields the sweep summary doesn't yet
    # use — lands in the per-seed CSV for future analysis.
    for k, v in res.items():
        if k == 'hier_diag_by_level':
            continue
        if isinstance(v, dict):
            # Flatten node-count dict: hier_nodes_by_level -> hier_nodes_level_0 etc.
            if k == 'hier_nodes_by_level':
                for level_key, count in v.items():
                    out[f'hier_nodes_{level_key}'] = int(count)
                out['hier_n_levels_used'] = int(len(v))
                out['hier_total_nodes']   = int(sum(v.values()))
            # All other dicts are skipped (already captured field-by-field below)
        else:
            out[k] = v

    # ── all-heldout metrics (every test rep, not just first) ──────
    # trial_df contains only first-heldout trials from the runner.
    # Re-derive all-heldout from the runner's meta if trial_df covers
    # it; otherwise fall back to first-heldout values.
    # The runner only returns first-heldout in trial_df, so
    # all-heldout requires us to re-examine the task.  We compute it
    # here from trial_df directly where is_first_heldout=True is all
    # we have; for test_reps=1 first==all so this is always correct.
    # When test_reps > 1, the runner would need to return all test
    # trials — for now we store NaN for all-heldout to keep the
    # contract clean and avoid silent errors.
    if test_reps == 1:
        # first-heldout IS all-heldout
        for suffix in ('acc', 'brier', 'logloss_reduction'):
            flat_key = f'flat_{"first_heldout" if suffix == "acc" else "heldout"}_{suffix}'
            hier_key = f'hier_{"first_heldout" if suffix == "acc" else "heldout"}_{suffix}'
            adv_key  = f'adv_{"first_heldout" if suffix == "acc" else "heldout"}_{suffix}'
            out[f'flat_all_heldout_{suffix}'] = out.get(flat_key, np.nan)
            out[f'hier_all_heldout_{suffix}'] = out.get(hier_key, np.nan)
            out[f'adv_all_heldout_{suffix}']  = out.get(adv_key, np.nan)
        out['flat_all_heldout_shape_contrast'] = out.get('flat_heldout_shape_contrast', np.nan)
        out['hier_all_heldout_shape_contrast'] = out.get('hier_heldout_shape_contrast', np.nan)
        out['adv_all_heldout_shape_contrast']  = out.get('adv_heldout_shape_contrast', np.nan)
    else:
        for key in ('flat_all_heldout_acc', 'hier_all_heldout_acc', 'adv_all_heldout_acc',
                    'flat_all_heldout_brier', 'hier_all_heldout_brier', 'adv_all_heldout_brier_reduction',
                    'flat_all_heldout_logloss', 'hier_all_heldout_logloss', 'adv_all_heldout_logloss_reduction',
                    'flat_all_heldout_shape_contrast', 'hier_all_heldout_shape_contrast',
                    'adv_all_heldout_shape_contrast'):
            out[key] = np.nan   # runner doesn't expose all-test rows yet

    # ── per-shape heldout accuracy ────────────────────────────────
    # Computed from trial_df which has shape, flat_correct, hier_correct.
    for s in [0, 1]:
        sub = trial_df[trial_df['shape'] == s] if len(trial_df) > 0 else pd.DataFrame()
        flat_acc_s = float(sub['flat_correct'].mean()) if len(sub) > 0 else np.nan
        hier_acc_s = float(sub['hier_correct'].mean()) if len(sub) > 0 else np.nan
        out[f'flat_heldout_shape{s}_acc'] = flat_acc_s
        out[f'hier_heldout_shape{s}_acc'] = hier_acc_s
        out[f'adv_heldout_shape{s}_acc']  = (hier_acc_s - flat_acc_s
                                              if not (np.isnan(hier_acc_s) or np.isnan(flat_acc_s))
                                              else np.nan)

    # ── AUC + balanced accuracy on held-out (novel-color) trials ──
    # Full-curve AUC needs the per-trial prediction scores, which exist in
    # trial_df here (before aggregation). AUC = P(rewarded-shape trial ranked
    # above other-shape trial); threshold-free, invariant to the compression
    # that length normalization induces. Balanced accuracy = (TPR+TNR)/2 at the
    # 0.5 boundary. Positive class = the rewarded shape.
    def _auc_and_balacc(prob_col):
        if len(trial_df) == 0 or prob_col not in trial_df.columns:
            return np.nan, np.nan
        rs = out.get('rewarded_shape', None)
        # rewarded_shape may not be set yet at this point; infer locally
        if rs is None or rs < 0:
            if 'reward' in trial_df.columns and (trial_df['reward'] == 1).any():
                rs_local = int(trial_df[trial_df['reward'] == 1]['shape'].mode()[0])
            else:
                return np.nan, np.nan
        else:
            rs_local = rs
        lab = (trial_df['shape'].values == rs_local).astype(int)   # 1 = rewarded shape
        scores = trial_df[prob_col].values
        if lab.sum() == 0 or lab.sum() == len(lab):
            return np.nan, np.nan
        try:
            auc = float(roc_auc_score(lab, scores))
        except Exception:
            auc = np.nan
        # balanced accuracy at 0.5: mean of per-class accuracy
        pred = (scores > 0.5).astype(int)
        tpr = float((pred[lab == 1] == 1).mean()) if (lab == 1).any() else np.nan
        tnr = float((pred[lab == 0] == 0).mean()) if (lab == 0).any() else np.nan
        bal = np.nanmean([tpr, tnr])
        return auc, bal

    flat_auc, flat_bal = _auc_and_balacc('flat_prob')
    hier_auc, hier_bal = _auc_and_balacc('hier_prob')
    out['flat_heldout_auc'] = flat_auc
    out['hier_heldout_auc'] = hier_auc
    out['adv_heldout_auc']  = (hier_auc - flat_auc
                               if not (np.isnan(hier_auc) or np.isnan(flat_auc)) else np.nan)
    out['flat_heldout_bal_acc'] = flat_bal
    out['hier_heldout_bal_acc'] = hier_bal
    out['adv_heldout_bal_acc']  = (hier_bal - flat_bal
                                   if not (np.isnan(hier_bal) or np.isnan(flat_bal)) else np.nan)

    # ── rewarded shape identity ───────────────────────────────────
    # Needed when reward_rule='random_balanced' to verify the advantage
    # isn't driven by one shape being systematically easier.
    # Infer from trial_df: the rewarded shape is the one where reward==1
    # and shape matches.
    if len(trial_df) > 0 and 'reward' in trial_df.columns:
        rewarded = trial_df[trial_df['reward'] == 1]['shape']
        out['rewarded_shape'] = int(rewarded.mode()[0]) if len(rewarded) > 0 else -1
    else:
        out['rewarded_shape'] = int(res.get('rewarded_shape', -1))

    # ── adv_train_acc (not in runner result) ─────────────────────
    flat_tr = out.get('flat_train_acc', np.nan)
    hier_tr = out.get('hier_train_acc', np.nan)
    out['adv_train_acc'] = (hier_tr - flat_tr
                            if not (np.isnan(hier_tr) or np.isnan(flat_tr))
                            else np.nan)

    return out


# ============================================================================
# PARALLEL ANALYSIS
# ============================================================================

def run_parallel_analysis(
    n_seeds,
    alpha,
    omega,
    n_particles,
    max_depth,
    max_children,
    length_normalize,
    tau,
    n_colors,
    n_train_colors,
    n_test_colors,
    train_reps,
    test_reps,
    shape_copies,
    color_copies,
    n_nuisance_features,
    blocked_train_by_shape,
    reward_rule,
    n_workers=None,
):
    """
    Run run_single_seed in parallel across n_seeds seeds.
    Returns a flat list of result dicts (one per seed).
    """
    if n_workers is None:
        n_workers = min(16, max(1, cpu_count() - 2))

    worker = partial(
        run_single_seed,
        n_colors=n_colors,
        n_train_colors=n_train_colors,
        n_test_colors=n_test_colors,
        train_reps=train_reps,
        test_reps=test_reps,
        shape_copies=shape_copies,
        color_copies=color_copies,
        n_nuisance_features=n_nuisance_features,
        blocked_train_by_shape=blocked_train_by_shape,
        reward_rule=reward_rule,
        n_particles=n_particles,
        alpha=alpha,
        omega=omega,
        max_depth=max_depth,
        max_children=max_children,
        length_normalize=length_normalize,
        tau=tau,

    )

    with Pool(n_workers) as pool:
        results = pool.map(worker, range(n_seeds))

    return results


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def _mean_adv(results, key):
    """Mean of a scalar key across seeds, ignoring NaN."""
    vals = pd.to_numeric(
        pd.Series([r.get(key, np.nan) for r in results]),
        errors='coerce'
    ).dropna()
    return float(vals.mean()) if len(vals) > 0 else np.nan


def _sem_adv(results, key):
    vals = pd.to_numeric(
        pd.Series([r.get(key, np.nan) for r in results]),
        errors='coerce'
    ).dropna()
    if len(vals) < 2:
        return np.nan
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def compute_sweep_scores(results):
    """
    Compute summary statistics for one parameter combination.

    Returns mean ± SE for every scalar metric stored by run_single_seed,
    plus composite optimisation scores.

    Composite scores
    ----------------
    score_probabilistic  = adv_heldout_brier_reduction
                         + 0.5 * adv_heldout_logloss_reduction
                         + adv_heldout_shape_contrast
        Primary ranking metric.  Continuous, threshold-independent.

    score_with_accuracy  = score_probabilistic
                         + 0.1 * adv_first_heldout_acc
        Tiebreaker that incorporates the binary accuracy advantage.

    All metrics that appear in any seed result dict are summarised —
    even ones not yet used for ranking — so the summary CSV contains
    everything needed for post-hoc analysis without re-running.
    """
    if not results:
        return {'n_seeds': 0, 'score_probabilistic': np.nan,
                'score_with_accuracy': np.nan}

    # Collect every numeric key that appears in at least one result
    all_keys = set()
    for r in results:
        for k, v in r.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                all_keys.add(k)
            elif isinstance(v, str):
                pass   # keep string fields below separately

    # String fields: take mode (most common value)
    string_keys = {'path_readout_mode', 'reward_rule'}
    out = {}
    for k in string_keys:
        vals = [r[k] for r in results if k in r and isinstance(r[k], str)]
        if vals:
            out[k] = max(set(vals), key=vals.count)

    # Numeric fields: mean and SE
    for k in sorted(all_keys):
        # Skip parameter identity columns — they're fixed within a combo
        if k in ('seed', 'alpha', 'omega',
                 #'level_decay', 
                 'n_particles', 'train_reps', 'test_reps',
                 'shape_copies', 'color_copies', 'blocked_train',
                 'n_colors', 'n_train_colors', 'n_test_colors',
                 'rewarded_shape'):
            # Store the fixed value once (all seeds share it except rewarded_shape)
            unique_vals = list({r[k] for r in results if k in r})
            if len(unique_vals) == 1:
                out[k] = unique_vals[0]
            continue
        out[f'{k}_mean'] = _mean_adv(results, k)
        out[f'{k}_sem']  = _sem_adv(results, k)

    # ── parameter identity (fixed within combo) ───────────────────
    for k in ('alpha', 'omega',
            #'level_decay',
              'n_particles', 'train_reps', 'test_reps',
              'shape_copies', 'color_copies', 'blocked_train',
              'n_colors', 'n_train_colors', 'n_test_colors'):
        vals = [r[k] for r in results if k in r]
        if vals:
            out[k] = vals[0]

    # ── composite scores ──────────────────────────────────────────
    out['score_probabilistic'] = (
          _nan_to_zero(out.get('adv_heldout_brier_reduction_mean'))
        + 0.5 * _nan_to_zero(out.get('adv_heldout_logloss_reduction_mean'))
        + _nan_to_zero(out.get('adv_heldout_shape_contrast_mean'))
    )
    out['score_with_accuracy'] = (
          out['score_probabilistic']
        + 0.1 * _nan_to_zero(out.get('adv_first_heldout_acc_mean'))
    )
    out['n_seeds'] = len(results)

    return out


def _nan_to_zero(v):
    """Convert NaN to 0 for composite score computation."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep(
    n_param_settings=50,
    n_seeds=30,
    alpha_range=(0.1, 1.0),
    omega_range=(0.05, 0.5),
    n_particles=200,
    max_depth=20,
    max_children=20,
    length_normalize=True,
    tau=1.0,
    n_colors=10,
    n_train_colors=8,
    n_test_colors=2,
    train_reps=5,
    test_reps=3,
    shape_copies=6,
    color_copies=1,
    n_nuisance_features=0,
    blocked_train_by_shape=False,
    reward_rule='random_balanced',
    n_workers=None,
    master_seed=13,
    optimization_objective='probabilistic',
):
    """
    Sweep over randomly-sampled (alpha, omega)

    Follows transfer_task.py pattern:
        - Random sample of (alpha, omega) pairs from the specified ranges
        - run_parallel_analysis for each combination
        - Print progress and running best after each combo

    Returns
    -------
    best_params : dict   Best parameter combination found
    sweep_results : list of dicts   One entry per parameter combination
    """
    log_progress("=" * 80)
    log_progress("SHAPE × MANY COLORS PARAMETER SWEEP")
    log_progress("=" * 80)

    # Build parameter combinations
    rng = np.random.RandomState(master_seed)
    alpha_vals = rng.uniform(alpha_range[0], alpha_range[1], size=n_param_settings)
    omega_vals = rng.uniform(omega_range[0], omega_range[1], size=n_param_settings)
    ao_pairs = list(zip(alpha_vals, omega_vals))

    all_combos = [
        (alpha, omega)
        for (alpha, omega) in ao_pairs
    ]

    total = len(all_combos)

    log_progress(f"  (alpha, omega) pairs : {n_param_settings}")
    log_progress(f"  Total combinations   : {total}")
    log_progress(f"  Seeds per combo      : {n_seeds}")
    log_progress(f"  Total model runs     : {total * n_seeds}")
    log_progress(f"  Optimization target  : {optimization_objective}")
    log_progress(f"  Shuffled training    : {not blocked_train_by_shape}")
    log_progress(f"  Reward rule          : {reward_rule}")

    sweep_results = []
    best_score = -np.inf
    best_params = None

    for idx, (alpha, omega) in enumerate(all_combos):
        log_progress(
            f"\n[{idx+1}/{total}] alpha={alpha:.3f}  omega={omega:.3f}  ")

        results = run_parallel_analysis(
            n_seeds=n_seeds,
            alpha=alpha,
            omega=omega,
            n_particles=n_particles,
            max_depth=max_depth,
            max_children=max_children,
            length_normalize=length_normalize,
            tau=tau,
            n_colors=n_colors,
            n_train_colors=n_train_colors,
            n_test_colors=n_test_colors,
            train_reps=train_reps,
            test_reps=test_reps,
            shape_copies=shape_copies,
            color_copies=color_copies,
            n_nuisance_features=n_nuisance_features,
            blocked_train_by_shape=blocked_train_by_shape,
            reward_rule=reward_rule,
            n_workers=n_workers,
        )

        scores = compute_sweep_scores(results)
        scores.update({
            'alpha': alpha, 'omega': omega,
            'seed_results': results,
        })
        sweep_results.append(scores)

        # Choose optimization score
        if optimization_objective == 'accuracy':
            opt_score = scores.get('adv_first_heldout_acc_mean', -np.inf)
            score_name = "heldout accuracy advantage"
        else:
            opt_score = scores.get('score_probabilistic', -np.inf)
            score_name = "probabilistic score"

        log_progress(
            f"  -> BrierRed={scores['adv_heldout_brier_reduction_mean']:+.4f} "
            f"(±{scores['adv_heldout_brier_reduction_sem']:.4f})  "
            f"LogRed={scores['adv_heldout_logloss_reduction_mean']:+.4f}  "
            f"ContrastAdv={scores['adv_heldout_shape_contrast_mean']:+.4f}  "
            f"AccAdv={scores['adv_first_heldout_acc_mean']:+.3f}"
        )
        log_progress(f"  -> {score_name}: {opt_score:.4f}")

        if opt_score > best_score:
            best_score = opt_score
            best_params = {
                'alpha': alpha, 'omega': omega}
            log_progress(f"  ✓ New best {score_name}!")

    return best_params, sweep_results


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_sweep_summary(sweep_results, optimization_objective='probabilistic'):
    """
    Print top-10 parameter combinations and the tree_mix ablation summary.
    """
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'seed_results'}
        for r in sweep_results
    ])

    score_col = ('adv_first_heldout_acc_mean'
                 if optimization_objective == 'accuracy'
                 else 'score_probabilistic')

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)

    print(f"\nTop 10 by {score_col}:")
    show = [
        'alpha', 'omega',
        'adv_heldout_brier_reduction_mean',
        'adv_heldout_logloss_reduction_mean',
        'adv_heldout_shape_contrast_mean',
        'adv_first_heldout_acc_mean',
        score_col,
        'n_seeds',
    ]
    show = [c for c in show if c in df.columns]
    print(df.nlargest(10, score_col)[show].to_string(index=False,
          float_format=lambda x: f'{x:.4f}'))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for Shape × Many Colors transfer task')

    # Sweep control
    parser.add_argument('--n_param_settings', type=int, default=200,
                        help='Number of random (alpha, omega) pairs to sample')
    parser.add_argument('--n_seeds', type=int, default=6,
                        help='Seeds per parameter combination')
    parser.add_argument('--master_seed', type=int, default=13)
    parser.add_argument('--optimize', type=str, default='probabilistic',
                        choices=['probabilistic', 'accuracy'],
                        help='Optimization objective')

    # Parameter ranges — alpha and omega are swept randomly
    parser.add_argument('--alpha_range', type=float, nargs=2, default=[0.1, 1.0])
    parser.add_argument('--omega_range', type=float, nargs=2, default=[0.05, 0.5])

    # Model config
    parser.add_argument('--n_particles', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--max_children', type=int, default=20)
    parser.add_argument('--no_length_normalize', action='store_true')
    parser.add_argument('--tau', type=float, default=1.0)

    # Task config
    parser.add_argument('--n_colors', type=int, default=10)
    parser.add_argument('--n_train_colors', type=int, default=8)
    parser.add_argument('--n_test_colors', type=int, default=2)
    parser.add_argument('--train_reps', type=int, default=10)
    parser.add_argument('--test_reps', type=int, default=3)
    parser.add_argument('--shape_copies', type=int, default=6)
    parser.add_argument('--color_copies', type=int, default=1)
    parser.add_argument('--nuisance_features', type=int, default=0)
    parser.add_argument('--shuffled_train', action='store_true')
    parser.add_argument('--reward_rule', type=str, default='random_balanced',
                        choices=['shape0', 'random_balanced'])

    # Infrastructure
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--out_prefix', type=str, default='shape_color_sweep')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving results to disk (default: always save).')

    args = parser.parse_args()

    log_progress("=" * 80)
    log_progress("SHAPE × MANY COLORS PARAMETER SWEEP")
    log_progress("=" * 80)
    log_progress(f"CPUs available: {cpu_count()}")
    log_progress(f"Optimization objective: {args.optimize}")

    best_params, sweep_results = run_parameter_sweep(
        n_param_settings=args.n_param_settings,
        n_seeds=args.n_seeds,
        alpha_range=tuple(args.alpha_range),
        omega_range=tuple(args.omega_range),
        n_particles=args.n_particles,
        max_depth=args.max_depth,
        max_children=args.max_children,
        length_normalize=not args.no_length_normalize,
        tau=args.tau,
        n_colors=args.n_colors,
        n_train_colors=args.n_train_colors,
        n_test_colors=args.n_test_colors,
        train_reps=args.train_reps,
        test_reps=args.test_reps,
        shape_copies=args.shape_copies,
        color_copies=args.color_copies,
        n_nuisance_features=args.nuisance_features,
        blocked_train_by_shape=not args.shuffled_train,
        reward_rule=args.reward_rule,
        n_workers=args.n_workers,
        master_seed=args.master_seed,
        optimization_objective=args.optimize,
    )

    print_sweep_summary(sweep_results, optimization_objective=args.optimize)

    log_progress("\n" + "=" * 80)
    log_progress(f"Best parameters: {best_params}")
    log_progress("=" * 80)

    if not args.no_save:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        # ── per-seed CSV: every scalar from every seed run ────────
        # This is the primary archival artefact. All future analyses
        # should be derivable from this file without re-running.
        all_seed_rows = []
        for r in sweep_results:
            for seed_res in r.get('seed_results', []):
                all_seed_rows.append(seed_res)
        if all_seed_rows:
            csv_fname = f"{args.out_prefix}_{date_str}_per_seed.csv"
            pd.DataFrame(all_seed_rows).to_csv(csv_fname, index=False)
            log_progress(f"Per-seed CSV saved → {csv_fname}")

        # ── summary CSV: one row per parameter combination ────────
        summary_rows = [
            {k: v for k, v in r.items() if k != 'seed_results'}
            for r in sweep_results
        ]
        summary_fname = f"{args.out_prefix}_{date_str}_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_fname, index=False)
        log_progress(f"Summary CSV saved → {summary_fname}")

        # ── pickle: best params + summary (no large arrays) ───────
        pkl_fname = f"{args.out_prefix}_{date_str}.pkl"
        with open(pkl_fname, 'wb') as fh:
            pickle.dump({
                'best_params': best_params,
                'summary':     summary_rows,
                'args':        vars(args),
            }, fh)
        log_progress(f"Pickle saved → {pkl_fname}")


if __name__ == '__main__':
    main()