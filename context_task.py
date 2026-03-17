#!/usr/bin/env python3
"""
Hierarchical vs Flat Model Comparison on Nested Temporal Task

Task Structure:
- SLOW level: Which rule applies? (shape-rule vs texture-rule)
  - Shape-rule: Shape determines reward (Circle/Triangle), texture irrelevant
  - Texture-rule: Texture determines reward (Dots/Stripes), shape irrelevant
  
- FAST level: Which value is rewarded?
  - In shape-rule contexts: Circle rewarded, then Triangle, alternating
  - In texture-rule contexts: Dots rewarded, then Stripes, alternating

"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import pickle
import datetime
import random
from scipy import stats

# Import model functions
from flat_model import *
from hier_model import *


# ============================================================================
# TASK GENERATOR
# ============================================================================

def generate_nested_temporal_task_v2(
    n_trials_per_slow_context=100,
    n_slow_contexts=4,
    trials_per_fast_switch=25,
    include_explicit_cues=False,
    include_fast_cues=False,
    balanced_stimuli=True,
    outcome_noise=0.02,
    seed=None
):
    """
    Generate hierarchical temporal task with alternating rule contexts.
    
    Task structure forces hierarchical learning:
    - Flat models cannot solve (stuck at ~50% chance)
    - Hierarchical models can discover slow/fast structure (~70-90%)
    
    Args:
        n_trials_per_slow_context: Trials per slow context (default: 100)
        n_slow_contexts: Number of slow contexts (default: 4, alternates shape/texture)
        trials_per_fast_switch: Trials per fast switch (default: 25)
        include_explicit_cues: Include observable slow-level rule cue (default: False)
        include_fast_cues: Include observable fast-level value cue (default: False)
        balanced_stimuli: Present all 4 stimuli in each block (default: True)
        outcome_noise: Outcome stochasticity (default: 0.02)
        seed: Random seed
    
    Returns:
        F: Feature matrix (features × trials), last row is outcome
        meta: Task metadata with trial info, block structure, etc.
        feedback_mask: Boolean mask (all True for this task)
    """
    rng = np.random.default_rng(seed)
    
    # Feature encoding:
    # [0]: Circle (1) or Triangle (0) 
    # [1]: Stripes (1) or Dots (0)
    # [2]: (optional) Slow cue: shape-rule (0) or texture-rule (1)
    # [3]: (optional) Fast cue: which value rewarded (0 or 1)
    # [last]: Outcome (reward/no-reward)
    
    n_shape_feats = 1
    n_texture_feats = 1
    n_slow_cue_feats = 1 if include_explicit_cues else 0
    n_fast_cue_feats = 1 if include_fast_cues else 0
    n_features = n_shape_feats + n_texture_feats + n_slow_cue_feats + n_fast_cue_feats
    outcome_idx = n_features
    
    # Define slow contexts (rule types)
    slow_contexts = {
        'shape_rule': {
            'id': 0,
            'rule_type': 'shape',
            'description': 'Shape determines reward (texture irrelevant)',
            'cue_value': 0
        },
        'texture_rule': {
            'id': 1,
            'rule_type': 'texture',
            'description': 'Texture determines reward (shape irrelevant)',
            'cue_value': 1
        }
    }
    
    # Alternating slow context sequence
    slow_sequence = ['shape_rule' if i % 2 == 0 else 'texture_rule' 
                     for i in range(n_slow_contexts)]
    
    # Generate trials
    all_trials = []
    slow_blocks = []
    fast_blocks = []
    trial_idx = 0
    fast_block_id = 0
    
    for slow_block_id, slow_context_name in enumerate(slow_sequence):
        slow_context = slow_contexts[slow_context_name]
        slow_block_start = trial_idx
        rule_type = slow_context['rule_type']
        
        # Number of fast switches within this slow context
        n_fast_switches = n_trials_per_slow_context // trials_per_fast_switch
        
        # Initialize rewarded value (0=Circle/Stripes, 1=Triangle/Dots)
        current_rewarded_value = rng.integers(0, 2)
        
        for fast_switch_id in range(n_fast_switches):
            fast_block_start = trial_idx
            
            # CRITICAL: Present ALL 4 stimulus combinations in each fast block
            # This ensures same stimulus → different outcome in different contexts
            if balanced_stimuli:
                # Each of 4 combos appears ~equally
                n_per_combo = trials_per_fast_switch // 4
                remainder = trials_per_fast_switch % 4
                
                stim_sequence = []
                for shape in [0, 1]:  # 0=Circle, 1=Triangle
                    for texture in [0, 1]:  # 0=Stripes, 1=Dots
                        count = n_per_combo + (1 if len(stim_sequence) < remainder else 0)
                        stim_sequence.extend([(shape, texture)] * count)
                
                rng.shuffle(stim_sequence)
            else:
                # Random stimulus presentations
                stim_sequence = [(rng.integers(0, 2), rng.integers(0, 2)) 
                               for _ in range(trials_per_fast_switch)]
            
            # Generate trials for this fast block
            for trial_in_fast_block, (shape, texture) in enumerate(stim_sequence):
                # Encode features
                features = np.zeros(n_features, dtype=int)
                features[0] = shape
                features[1] = texture
                
                # Optional cues
                if include_explicit_cues:
                    features[2] = slow_context['cue_value']
                if include_fast_cues:
                    cue_idx = n_shape_feats + n_texture_feats + n_slow_cue_feats
                    features[cue_idx] = current_rewarded_value
                
                # Determine outcome based on current rule and rewarded value
                if rule_type == 'shape':
                    outcome = int(shape == current_rewarded_value)
                else:  # texture rule
                    outcome = int(texture == current_rewarded_value)
                
                # Add noise
                if rng.random() < outcome_noise:
                    outcome = 1 - outcome
                
                # Determine if this is a switch trial
                is_slow_switch = (slow_block_id > 0 and 
                                trial_in_fast_block == 0 and 
                                fast_switch_id == 0)
                is_fast_switch = (fast_switch_id > 0 and 
                                trial_in_fast_block == 0)
                
                all_trials.append({
                    'trial_idx': trial_idx,
                    'slow_block_id': slow_block_id,
                    'fast_block_id': fast_block_id,
                    'slow_context_name': slow_context_name,
                    'rule_type': rule_type,
                    'trial_in_slow_block': trial_idx - slow_block_start,
                    'trial_in_fast_block': trial_in_fast_block,
                    'shape': shape,
                    'texture': texture,
                    'rewarded_value': current_rewarded_value,
                    'rewarded_feature': 'shape' if rule_type == 'shape' else 'texture',
                    'features': features,
                    'outcome': outcome,
                    'is_slow_switch': is_slow_switch,
                    'is_fast_switch': is_fast_switch,
                })
                
                trial_idx += 1
            
            # Record fast block metadata
            fast_blocks.append({
                'fast_block_id': fast_block_id,
                'slow_block_id': slow_block_id,
                'rule_type': rule_type,
                'rewarded_value': current_rewarded_value,
                'start': fast_block_start,
                'end': trial_idx - 1,
                'n_trials': trials_per_fast_switch
            })
            
            fast_block_id += 1
            
            # Alternate rewarded value for next fast block
            current_rewarded_value = 1 - current_rewarded_value
        
        # Record slow block metadata
        slow_blocks.append({
            'slow_block_id': slow_block_id,
            'context_name': slow_context_name,
            'rule_type': rule_type,
            'start': slow_block_start,
            'end': trial_idx - 1,
            'n_trials': trial_idx - slow_block_start,
            'n_fast_switches': n_fast_switches
        })
    
    n_trials = len(all_trials)
    
    # Create feature matrix
    F = np.zeros((n_features + 1, n_trials), dtype=int)
    for trial in all_trials:
        t = trial['trial_idx']
        F[:n_features, t] = trial['features']
        F[outcome_idx, t] = trial['outcome']
    
    # Create metadata
    meta = {
        'task_type': 'nested_temporal_v2',
        'n_trials': n_trials,
        'outcome_idx': outcome_idx,
        'n_features': n_features,
        'n_slow_contexts': n_slow_contexts,
        'trials_per_slow_context': n_trials_per_slow_context,
        'trials_per_fast_switch': trials_per_fast_switch,
        'include_explicit_cues': include_explicit_cues,
        'slow_contexts': slow_contexts,
        'slow_sequence': slow_sequence,
        'slow_blocks': slow_blocks,
        'fast_blocks': fast_blocks,
        'trials': all_trials,
        'feature_structure': {
            'shape_index': 0,
            'texture_index': 1,
            'slow_cue_index': 2 if include_explicit_cues else None,
            'fast_cue_index': (2 + n_slow_cue_feats) if include_fast_cues else None
        }
    }
    
    feedback_mask = np.ones((n_features + 1, n_trials), dtype=bool)
    
    return F, meta, feedback_mask


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_progress(message, flush=True):
    """Print message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=flush)


def _majority_id(particle_assignments):
    """Get majority-voted cluster/node ID from particle assignments."""
    if len(particle_assignments) == 0:
        return -1
    valid = particle_assignments[particle_assignments >= 0]
    if len(valid) == 0:
        return -1
    counts = np.bincount(valid.astype(int))
    return int(np.argmax(counts))


# ============================================================================
# HIERARCHICAL LEVEL-SPECIFIC PREDICTIONS
# ============================================================================

def get_hierarchical_predictions_by_level(paths_hier, true_outcomes, level=0):
    """
    Get outcome predictions from a specific hierarchical level.
    
    Args:
        paths_hier: Particle paths (nParticles, nTrials, max_depth)
        true_outcomes: True outcomes (nTrials,)
        level: Which tree level to use
    
    Returns:
        predictions: Probabilistic predictions (nTrials,) in [0, 1]
        binary_predictions: Binary predictions (nTrials,)
    """
    nParticles, nTrials, max_depth = paths_hier.shape
    
    if level >= max_depth:
        return np.zeros(nTrials), np.zeros(nTrials)
    
    # Get majority node at this level for each trial
    level_assignments = np.array([
        _majority_id(paths_hier[:, t, level]) for t in range(nTrials)
    ])
    
    # Compute empirical reward rate per node
    node_reward_rates = {}
    for t in range(nTrials):
        node = level_assignments[t]
        if node >= 0:
            if node not in node_reward_rates:
                node_reward_rates[node] = []
            node_reward_rates[node].append(true_outcomes[t])
    
    # Convert to probabilities
    node_probs = {}
    for node, outcomes in node_reward_rates.items():
        node_probs[node] = np.mean(outcomes)
    
    # Generate predictions
    predictions = np.zeros(nTrials)
    for t in range(nTrials):
        node = level_assignments[t]
        if node >= 0 and node in node_probs:
            predictions[t] = node_probs[node]
        else:
            predictions[t] = 0.5  # Default to chance
    
    binary_predictions = (predictions > 0.5).astype(int)
    
    return predictions, binary_predictions


def get_hierarchical_ensemble_predictions(paths_hier, true_outcomes, levels=[0, 1, 2]):
    """
    Ensemble predictions across multiple hierarchical levels via majority vote.
    
    Returns:
        ensemble_predictions_prob: Probabilistic predictions (average across levels)
        ensemble_predictions_binary: Binary predictions (majority vote)
        predictions_by_level: Dict of predictions per level
    """
    predictions_by_level_prob = {}
    predictions_by_level_binary = {}
    
    for level in levels:
        prob, binary = get_hierarchical_predictions_by_level(
            paths_hier, true_outcomes, level
        )
        predictions_by_level_prob[level] = prob
        predictions_by_level_binary[level] = binary
    
    # Probabilistic ensemble (average)
    if len(predictions_by_level_prob) > 0:
        all_probs = np.array(list(predictions_by_level_prob.values()))
        ensemble_prob = np.mean(all_probs, axis=0)
    else:
        ensemble_prob = np.zeros(len(true_outcomes))
    
    # Binary ensemble (majority vote)
    if len(predictions_by_level_binary) > 0:
        all_binary = np.array(list(predictions_by_level_binary.values()))
        ensemble_binary = (np.sum(all_binary, axis=0) > (len(all_binary) / 2)).astype(int)
    else:
        ensemble_binary = np.zeros(len(true_outcomes), dtype=int)
    
    return ensemble_prob, ensemble_binary, predictions_by_level_prob


def compute_level_specific_accuracies(paths_hier, meta, levels=[0, 1, 2]):
    """
    Compute accuracy for each hierarchical level and ensemble.
    
    Returns:
        Dictionary with accuracy by level and context type, PLUS predictions
    """
    trials = meta['trials']
    true_outcomes = np.array([t['outcome'] for t in trials])
    
    results = {}
    
    for level in levels:
        level_preds_prob, level_preds_binary = get_hierarchical_predictions_by_level(
            paths_hier, true_outcomes, level
        )
        level_acc = float(np.mean(level_preds_binary == true_outcomes))
        
        # Accuracy by rule type
        shape_trials = [i for i, t in enumerate(trials) 
                       if t.get('slow_context_name', '') == 'shape_rule']
        texture_trials = [i for i, t in enumerate(trials) 
                         if t.get('slow_context_name', '') == 'texture_rule']
        
        shape_acc = float(np.mean(level_preds_binary[shape_trials] == true_outcomes[shape_trials])) if shape_trials else 0.0
        texture_acc = float(np.mean(level_preds_binary[texture_trials] == true_outcomes[texture_trials])) if texture_trials else 0.0
        
        results[f'level_{level}'] = {
            'overall_acc': level_acc,
            'shape_context_acc': shape_acc,
            'texture_context_acc': texture_acc,
            'predictions_prob': level_preds_prob,  # NEW: probabilistic
            'predictions_binary': level_preds_binary  # NEW: binary
        }
    
    # Ensemble
    ensemble_prob, ensemble_binary, preds_by_level = get_hierarchical_ensemble_predictions(
        paths_hier, true_outcomes, levels
    )
    ensemble_acc = float(np.mean(ensemble_binary == true_outcomes))
    
    shape_trials = [i for i, t in enumerate(trials) if t.get('slow_context_name', '') == 'shape_rule']
    texture_trials = [i for i, t in enumerate(trials) if t.get('slow_context_name', '') == 'texture_rule']
    
    shape_acc = float(np.mean(ensemble_binary[shape_trials] == true_outcomes[shape_trials])) if shape_trials else 0.0
    texture_acc = float(np.mean(ensemble_binary[texture_trials] == true_outcomes[texture_trials])) if texture_trials else 0.0
    
    results['ensemble'] = {
        'overall_acc': ensemble_acc,
        'shape_context_acc': shape_acc,
        'texture_context_acc': texture_acc,
        'predictions_prob': ensemble_prob,  # NEW: probabilistic
        'predictions_binary': ensemble_binary  # NEW: binary
    }
    
    return results


# ============================================================================
# ANALYSIS METRICS
# ============================================================================

def compute_switch_cost(predictions_prob, meta):
    """
    Compute switch cost: accuracy drop at context switches.
    Uses probabilistic predictions (0-1) instead of binary.
    
    Returns dict with switch_cost, switch_trial_acc, non_switch_trial_acc
    """
    trials = meta['trials']
    true_outcomes = np.array([t['outcome'] for t in trials])
    predicted_binary = (predictions_prob > 0.5).astype(int)
    correct = (predicted_binary == true_outcomes).astype(float)
    
    # Slow switches are the main context changes
    switch_trials_idx = [i for i, t in enumerate(trials) if t.get('is_slow_switch', False)]
    non_switch_trials_idx = [i for i, t in enumerate(trials) if not t.get('is_slow_switch', False)]
    
    switch_acc = float(np.mean(correct[switch_trials_idx])) if switch_trials_idx else 0.0
    non_switch_acc = float(np.mean(correct[non_switch_trials_idx])) if non_switch_trials_idx else 0.0
    
    return {
        'switch_trial_acc': switch_acc,
        'non_switch_trial_acc': non_switch_acc,
        'switch_cost': non_switch_acc - switch_acc,
        'n_switch_trials': len(switch_trials_idx)
    }


def compute_cluster_reuse(particles, paths_hier, meta):
    """
    Measure representational compression.
    
    Returns cluster counts for flat and hierarchical models
    """
    n_particles, n_trials = particles.shape
    
    # Flat model clusters
    cluster_assignments = np.array([_majority_id(particles[:, t]) for t in range(n_trials)])
    total_flat_clusters = len(np.unique(cluster_assignments[cluster_assignments >= 0]))
    
    # Hierarchical nodes
    if paths_hier is not None:
        nParticles, nTrials, max_depth = paths_hier.shape
        hier_nodes_per_level = {}
        for L in range(max_depth):
            level_assignments = np.array([_majority_id(paths_hier[:, t, L]) 
                                         for t in range(nTrials)])
            n_nodes = len(np.unique(level_assignments[level_assignments >= 0]))
            hier_nodes_per_level[f'level_{L}'] = int(n_nodes)
        total_hier_nodes = sum(hier_nodes_per_level.values())
    else:
        hier_nodes_per_level = {}
        total_hier_nodes = 0
    
    return {
        'flat_total_clusters': int(total_flat_clusters),
        'hier_nodes_per_level': hier_nodes_per_level,
        'hier_total_nodes': int(total_hier_nodes),
        'compression_advantage': float(total_flat_clusters - total_hier_nodes)
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_single_seed(seed, n_trials_per_context, n_blocks, alpha, omega, include_cue):
    """
    Run one seed: generate task, train both models, compute metrics.
    
    Returns dict with ALL metrics and raw data
    """
    # Generate task
    F, meta, fb = generate_nested_temporal_task_v2(
        n_trials_per_slow_context=n_trials_per_context,
        n_slow_contexts=n_blocks,
        trials_per_fast_switch=n_trials_per_context // 4,
        include_explicit_cues=include_cue,
        seed=seed
    )
    
    n_trials = F.shape[1]
    outcome_idx = meta['outcome_idx']
    true_outcomes = F[outcome_idx, :]
    
    # Train flat model
    np.random.seed(seed)
    p_flat, _, rEst_flat, _ = one_layer_inference_loop(
        nTrials=n_trials,
        nParticles=200,
        nFeatures=F.shape[0],
        regime=omega,
        alpha=alpha,
        f=F,
        outcome_idx_per_trial=np.full(n_trials, outcome_idx, dtype=int),
        feedback_mask=fb[outcome_idx, :],
        random_seed=seed,
    )
    
    # Train hierarchical model
    np.random.seed(seed)
    p_hier, _, rEst_hier, _, _, paths_hier, _ = full_hier_inference_loop(
        nTrials=n_trials,
        nParticles=200,
        nFeatures=F.shape[0],
        alpha=alpha,
        omega=omega,
        f=F,
        max_depth=20,
        max_children=20,
        outcome_idx=outcome_idx,
        feedback_mask=fb,
        random_seed=seed,
    )
    
    # Compute hierarchical level-specific accuracies
    level_accs = compute_level_specific_accuracies(paths_hier, meta, levels=[0, 1, 2])
    
    # Use ensemble for overall metrics
    ensemble_pred_prob = level_accs['ensemble']['predictions_prob']
    ensemble_pred_binary = level_accs['ensemble']['predictions_binary']
    
    # Compute overall accuracies
    flat_acc = float(np.mean((rEst_flat > 0.5) == true_outcomes))
    hier_acc = level_accs['ensemble']['overall_acc']
    
    # Compute switch cost (using probabilistic predictions)
    flat_switch = compute_switch_cost(rEst_flat, meta)
    hier_switch = compute_switch_cost(ensemble_pred_prob, meta)
    
    # Compute compression
    reuse = compute_cluster_reuse(p_flat, paths_hier, meta)
    
    # Assemble results with EVERYTHING
    result = {
        # Seed info
        'seed': seed,
        'n_trials': n_trials,
        
        # Parameters
        'alpha': alpha,
        'omega': omega,
        'include_cue': include_cue,
        
        # Level 0 metrics
        'hier_level_0_overall': level_accs['level_0']['overall_acc'],
        'hier_level_0_shape': level_accs['level_0']['shape_context_acc'],
        'hier_level_0_texture': level_accs['level_0']['texture_context_acc'],
        'hier_level_0_nodes': reuse['hier_nodes_per_level'].get('level_0', 0),
        
        # Level 1 metrics (if exists)
        'hier_level_1_overall': level_accs.get('level_1', {}).get('overall_acc', np.nan),
        'hier_level_1_shape': level_accs.get('level_1', {}).get('shape_context_acc', np.nan),
        'hier_level_1_texture': level_accs.get('level_1', {}).get('texture_context_acc', np.nan),
        'hier_level_1_nodes': reuse['hier_nodes_per_level'].get('level_1', 0),
        
        # Level 2 metrics (if exists)
        'hier_level_2_overall': level_accs.get('level_2', {}).get('overall_acc', np.nan),
        'hier_level_2_shape': level_accs.get('level_2', {}).get('shape_context_acc', np.nan),
        'hier_level_2_texture': level_accs.get('level_2', {}).get('texture_context_acc', np.nan),
        'hier_level_2_nodes': reuse['hier_nodes_per_level'].get('level_2', 0),
        
        # Ensemble metrics
        'hier_ensemble_overall': hier_acc,
        'hier_ensemble_shape': level_accs['ensemble']['shape_context_acc'],
        'hier_ensemble_texture': level_accs['ensemble']['texture_context_acc'],
        
        # Overall accuracies
        'flat_overall_acc': flat_acc,
        'hier_overall_acc': hier_acc,
        'overall_acc_advantage': hier_acc - flat_acc,
        
        # Switch costs
        'flat_switch_cost': flat_switch['switch_cost'],
        'hier_switch_cost': hier_switch['switch_cost'],
        'switch_cost_reduction': flat_switch['switch_cost'] - hier_switch['switch_cost'],
        'flat_switch_trial_acc': flat_switch['switch_trial_acc'],
        'hier_switch_trial_acc': hier_switch['switch_trial_acc'],
        
        # Compression
        'flat_n_clusters': reuse['flat_total_clusters'],
        'hier_n_nodes': reuse['hier_total_nodes'],
        'compression_advantage': reuse['compression_advantage'],
        'hier_nodes_per_level': reuse['hier_nodes_per_level'],
        
        # Raw paths and particles
        'paths_hier': paths_hier.tolist(),
        'paths_flat': p_flat.tolist(),
        
        # Trial-by-trial data
        'trial_data': {
            'true_outcomes': true_outcomes.tolist(),
            'rEst_flat': rEst_flat.tolist(),
            
            # Hierarchical predictions at each level (PROBABILISTIC)
            'rEst_level_0_prob': level_accs['level_0']['predictions_prob'].tolist(),
            'rEst_level_1_prob': level_accs.get('level_1', {}).get('predictions_prob', []),
            'rEst_level_2_prob': level_accs.get('level_2', {}).get('predictions_prob', []),
            'rEst_hier_ensemble_prob': ensemble_pred_prob.tolist(),
            
            # Hierarchical predictions at each level (BINARY)
            'rEst_level_0_binary': level_accs['level_0']['predictions_binary'].tolist(),
            'rEst_level_1_binary': level_accs.get('level_1', {}).get('predictions_binary', []),
            'rEst_level_2_binary': level_accs.get('level_2', {}).get('predictions_binary', []),
            'rEst_hier_ensemble_binary': ensemble_pred_binary.tolist(),
            
            # Task metadata
            'meta': meta
        }
    }
    
    return result

def run_parallel_analysis(
    n_trials_per_context=50,
    n_blocks=4,
    n_seeds=24,
    alpha=4.0,
    omega=0.1,
    include_cue=False,
    n_workers=None,
    master_seed=42
):
    """
    Run analysis across multiple seeds in parallel.
    
    Returns:
        results: List of result dicts
        seed_sequence: List of seeds used
    """
    # Set master seed for reproducibility
    np.random.seed(master_seed)
    random.seed(master_seed)
    seed_sequence = np.random.randint(0, 100000, size=n_seeds).tolist()
    
    if n_workers is None:
        n_workers = min(16, max(1, cpu_count() - 2))
    
    log_progress("="*80)
    log_progress(f"Starting nested temporal task analysis with {n_workers} workers")
    log_progress(f"Parameters: alpha={alpha:.3f}, omega={omega:.3f}")
    log_progress(f"Master seed: {master_seed}")
    log_progress(f"Task: {n_blocks} slow contexts, {n_trials_per_context} trials each")
    log_progress(f"Explicit cue: {include_cue}")
    log_progress(f"Total seeds: {n_seeds}")
    log_progress("="*80)
    
    start = time.time()
    
    worker_func = partial(
        run_single_seed,
        n_trials_per_context=n_trials_per_context,
        n_blocks=n_blocks,
        alpha=alpha,
        omega=omega,
        include_cue=include_cue
    )
    
    results = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_func, seed_sequence)):
            results.append(result)
            
            # Progress reporting
            elapsed = time.time() - start
            avg_time = elapsed / (i + 1)
            eta = avg_time * (n_seeds - i - 1)
            
            hier_acc = result['hier_ensemble_overall']
            flat_acc = result['flat_overall_acc']
            log_progress(
                f"  [{i+1}/{n_seeds}] Seed {result['seed']}: "
                f"Flat={flat_acc:.1%}, Hier={hier_acc:.1%}, "
                f"Δ={hier_acc-flat_acc:+.1%} | ETA: {eta:.0f}s"
            )
    
    # Sort by seed for consistency
    results.sort(key=lambda x: x['seed'])
    
    elapsed = time.time() - start
    log_progress(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return results, seed_sequence


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results):
    """Print comprehensive summary of results."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("NESTED TEMPORAL TASK RESULTS")
    print("="*80)
    
    # Level-specific performance
    print(f"\n{'HIERARCHICAL LEVEL SPECIALIZATION'}")
    print("="*80)
    
    for level in [0, 1, 2]:
        # Check if this level exists in results
        if f'hier_level_{level}_overall' not in df.columns:
            continue
            
        overall = df[f'hier_level_{level}_overall'].mean()
        shape = df[f'hier_level_{level}_shape'].mean()
        texture = df[f'hier_level_{level}_texture'].mean()
        nodes = df[f'hier_level_{level}_nodes'].mean()
        
        print(f"\nLevel {level}:")
        print(f"  Overall:         {overall:.1%}")
        print(f"  Shape contexts:  {shape:.1%}")
        print(f"  Texture contexts: {texture:.1%}")
        print(f"  Avg nodes:       {nodes:.1f}")
    
    print(f"\nEnsemble (all levels):")
    print(f"  Overall:        {df['hier_ensemble_overall'].mean():.1%}")
    print(f"  Shape contexts:  {df['hier_ensemble_shape'].mean():.1%}")
    print(f"  Texture contexts: {df['hier_ensemble_texture'].mean():.1%}")
    
    # Overall accuracy
    print(f"\n{'OVERALL ACCURACY'}")
    print("="*80)
    print(f"Flat:         {df['flat_overall_acc'].mean():.1%} ± {df['flat_overall_acc'].std():.1%}")
    print(f"Hierarchical: {df['hier_overall_acc'].mean():.1%} ± {df['hier_overall_acc'].std():.1%}")
    print(f"Advantage:    {df['overall_acc_advantage'].mean():+.1%}")
    
    # Switch cost
    print(f"\n{'SWITCH COST'}")
    print("="*80)
    print(f"Flat switch cost:  {df['flat_switch_cost'].mean():.1%}")
    print(f"Hier switch cost:  {df['hier_switch_cost'].mean():.1%}")
    print(f"Reduction:         {df['switch_cost_reduction'].mean():+.1%}")
    print(f"\nSwitch trial accuracy:")
    print(f"  Flat: {df['flat_switch_trial_acc'].mean():.1%}")
    print(f"  Hier: {df['hier_switch_trial_acc'].mean():.1%}")
    
    # Compression
    print(f"\n{'REPRESENTATIONAL COMPRESSION'}")
    print("="*80)
    print(f"Flat clusters:    {df['flat_n_clusters'].mean():.1f} ± {df['flat_n_clusters'].std():.1f}")
    print(f"Hier total nodes: {df['hier_n_nodes'].mean():.1f} ± {df['hier_n_nodes'].std():.1f}")
    print(f"Compression:      {df['compression_advantage'].mean():+.1f} fewer nodes")
    
    # Statistical tests
    print(f"\n{'STATISTICAL TESTS'}")
    print("="*80)
    
    t_stat, p_val = stats.ttest_1samp(df['overall_acc_advantage'], 0)
    print(f"Overall accuracy: t({len(df)-1}) = {t_stat:.2f}, p = {p_val:.4f}")
    
    t_stat, p_val = stats.ttest_1samp(df['switch_cost_reduction'], 0)
    print(f"Switch cost:      t({len(df)-1}) = {t_stat:.2f}, p = {p_val:.4f}")
    
    t_stat, p_val = stats.ttest_1samp(df['compression_advantage'], 0)
    print(f"Compression:      t({len(df)-1}) = {t_stat:.2f}, p = {p_val:.4f}")


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep(
    n_param_settings=20,
    n_seeds=12,
    alpha_range=(0.01, 10.0),
    omega_range=(0.01, 3.0),
    optimization_objective='overall_accuracy',
    master_seed=42
):
    """
    Test multiple parameter combinations to find optimal alpha/omega.
    
    Args:
        n_param_settings: Number of parameter combinations to test
        n_seeds: Seeds per parameter combination
        alpha_range: (min, max) for alpha
        omega_range: (min, max) for omega
        optimization_objective: 'overall_accuracy' or 'switch_cost_reduction'
        master_seed: Master seed for reproducibility
    
    Returns:
        best_alpha: Optimal alpha
        best_omega: Optimal omega
        sweep_results: List of all tested combinations
    """
    # Set master seed
    np.random.seed(master_seed)
    random.seed(master_seed)
    
    log_progress("="*80)
    log_progress("PARAMETER SWEEP")
    log_progress("="*80)
    log_progress(f"Master seed: {master_seed}")
    log_progress(f"Optimization objective: {optimization_objective}")
    log_progress(f"Testing {n_param_settings} parameter combinations")
    
    # Generate parameter combinations (deterministic)
    param_combos = [(np.random.uniform(alpha_range[0], alpha_range[1]),
                     np.random.uniform(omega_range[0], omega_range[1]))
                    for _ in range(n_param_settings)]
    
    sweep_results = []
    best_score = -np.inf
    best_params = None
    
    for idx, (alpha, omega) in enumerate(param_combos):
        log_progress(f"\n[{idx+1}/{n_param_settings}] Testing alpha={alpha:.3f}, omega={omega:.3f}")
        
        # Run analysis with these parameters
        results, seed_seq = run_parallel_analysis(
            n_trials_per_context=50,
            n_blocks=4,
            n_seeds=n_seeds,
            alpha=alpha,
            omega=omega,
            include_cue=False,
            master_seed=master_seed + idx  # Different seed per combo
        )
        
        df = pd.DataFrame(results)
        
        # Compute metrics
        overall_acc = df['hier_overall_acc'].mean()
        switch_cost_reduction = df['switch_cost_reduction'].mean()
        compression = df['compression_advantage'].mean()
        
        sweep_results.append({
            'alpha': alpha,
            'omega': omega,
            'overall_acc': overall_acc,
            'flat_overall_acc': df['flat_overall_acc'].mean(),
            'switch_cost_reduction': switch_cost_reduction,
            'compression': compression,
            'hier_switch_cost': df['hier_switch_cost'].mean(),
            'flat_switch_cost': df['flat_switch_cost'].mean(),
            'level_0_acc': df['hier_level_0_overall'].mean(),
            'level_1_acc': df['hier_level_1_overall'].mean(),
            'level_2_acc': df['hier_level_2_overall'].mean(),
            'ensemble_acc': df['hier_ensemble_overall'].mean(),
            'results': results,  # Full results for this combo
            'seed_sequence': seed_seq
        })
        
        # Determine score based on objective
        if optimization_objective == 'overall_accuracy':
            opt_score = overall_acc
        elif optimization_objective == 'switch_cost_reduction':
            opt_score = switch_cost_reduction
        elif optimization_objective == 'compression':
            opt_score = compression
        else:
            raise ValueError(f"Unknown objective: {optimization_objective}")
        
        log_progress(f"  Overall accuracy:      {overall_acc:.1%}")
        log_progress(f"  Switch cost reduction: {switch_cost_reduction:+.1%}")
        log_progress(f"  Compression:           {compression:+.1f}")
        log_progress(f"  Optimization score:    {opt_score:.3f}")
        
        if opt_score > best_score:
            best_score = opt_score
            best_params = (alpha, omega)
            log_progress(f"  ✓ New best!")
    
    # Save sweep results
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'param_sweep_{optimization_objective}_{date_str}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'best_alpha': best_params[0],
            'best_omega': best_params[1],
            'optimization_objective': optimization_objective,
            'sweep_results': sweep_results,
            'master_seed': master_seed,
            'timestamp': datetime.datetime.now()
        }, f)
    
    log_progress(f"\n✓ Sweep results saved to {filename}")
    log_progress(f"✓ Best parameters: alpha={best_params[0]:.3f}, omega={best_params[1]:.3f}")
    
    return best_params[0], best_params[1], sweep_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical vs Flat Model Comparison')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sweep'],
                       help='Run mode: single analysis or parameter sweep')
    parser.add_argument('--n_seeds', type=int, default=6,
                       help='Number of seeds to run')
    parser.add_argument('--alpha', type=float, default=8.130,
                       help='Alpha parameter (CRP concentration)')
    parser.add_argument('--omega', type=float, default=0.236,
                       help='Omega parameter (stickiness)')
    parser.add_argument('--cue', action='store_true',
                       help='Include explicit context cue')
    parser.add_argument('--master_seed', type=int, default=13,
                       help='Master random seed for reproducibility')
    parser.add_argument('--optimize', type=str, default='overall_accuracy',
                       choices=['overall_accuracy', 'switch_cost_reduction', 'compression'],
                       help='Optimization objective for parameter sweep')
    parser.add_argument('--n_params', type=int, default=5,
                       help='Number of parameter combinations to test in sweep')
    
    args = parser.parse_args()
    
    log_progress("="*80)
    log_progress("NESTED TEMPORAL TASK: HIERARCHICAL VS FLAT")
    log_progress("="*80)
    
    if args.mode == 'single':
        # Single analysis
        results, seed_seq = run_parallel_analysis(
            n_trials_per_context=100,
            n_blocks=10,
            n_seeds=args.n_seeds,
            alpha=args.alpha,
            omega=args.omega,
            include_cue=args.cue,
            master_seed=args.master_seed
        )
        
        # Print summary
        print_summary(results)
        
        # Save results
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'nested_temporal_results_{date_str}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': results,
                'seed_sequence': seed_seq,
                'master_seed': args.master_seed,
                'alpha': args.alpha,
                'omega': args.omega,
                'timestamp': datetime.datetime.now()
            }, f)
        log_progress(f"\n✓ Results saved to {filename}")
    
    else:
        # Parameter sweep
        best_alpha, best_omega, sweep_results = run_parameter_sweep(
            n_param_settings=args.n_params,
            n_seeds=args.n_seeds,
            optimization_objective=args.optimize,
            master_seed=args.master_seed
        )
        
        # Print summary of best parameters
        print(f"\n{'='*80}")
        print("BEST PARAMETERS FOUND")
        print("="*80)
        print(f"Alpha: {best_alpha:.3f}")
        print(f"Omega: {best_omega:.3f}")
        print(f"Objective: {args.optimize}")