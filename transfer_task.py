#!/usr/bin/env python3
"""
COMPLETE Parameter Optimization Script - Outcome Prediction + Transfer Analysis


"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import pickle
import sys
import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Import model functions
from flat_model import *
from hier_model import *

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_progress(message, flush=True):
    """Print with timestamp and flush immediately"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=flush)


def _majority_id(particle_assignments):
    """
    Get majority-voted cluster/node ID from particle assignments.
    Returns -1 if no valid assignments exist.
    """
    if len(particle_assignments) == 0:
        return -1
    valid = particle_assignments[particle_assignments >= 0]
    if len(valid) == 0:
        return -1
    counts = np.bincount(valid.astype(int))
    return int(np.argmax(counts))


def compute_transfer_metrics_with_f1(predictions, true_labels, positive_label=0):
    """
    Compute transfer accuracy, precision, recall, and F1 score.
    
    Args:
        predictions: Binary predictions (0=same category, 1=different)
        true_labels: True category labels
        positive_label: Which label to treat as "positive class" (default: 0)
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, and confusion matrix
    """
    # Create binary masks
    true_positive_mask = (true_labels == positive_label)
    pred_positive_mask = (predictions == positive_label)
    
    # Confusion matrix elements
    TP = int(np.sum(pred_positive_mask & true_positive_mask))
    FP = int(np.sum(pred_positive_mask & ~true_positive_mask))
    FN = int(np.sum(~pred_positive_mask & true_positive_mask))
    TN = int(np.sum(~pred_positive_mask & ~true_positive_mask))
    
    # Metrics
    accuracy = (TP + TN) / len(predictions) if len(predictions) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


# ============================================================================
# TASK GENERATION
# ============================================================================

def generate_scalable_hierarchical_task_quiet(n_levels=3, trials_per_context=40, seed=None):
    """
    Generate scalable hierarchical task with n_levels of structure.
    Silent version - no print statements.
    
    Args:
        n_levels: Number of hierarchical levels (2-5)
        trials_per_context: Number of trials per context
        seed: Random seed for reproducibility
    
    Returns:
        F: Feature matrix (features x trials)
        meta: Metadata dictionary with true factors
        feedback_mask: Boolean mask for feedback availability
    """
    rng = np.random.default_rng(seed)
    
    # Calculate feature dimensions based on task structure
    n_top_levels = max(0, n_levels - 2)
    n_species_feats = 2 if n_levels >= 2 else 0
    n_color_feats = 4 if n_levels >= 1 else 0
    
    total_features = n_top_levels + n_species_feats + n_color_feats
    outcome_idx = total_features
    
    # Generate all possible contexts recursively
    def generate_all_contexts(n_levels):
        if n_levels == 1:
            return [[c] for c in range(4)]
        elif n_levels == 2:
            contexts = []
            for species in [0, 1]:
                for color in range(4):
                    contexts.append([color, species])
            return contexts
        else:
            sub_contexts = generate_all_contexts(n_levels - 1)
            contexts = []
            for top_val in [0, 1]:
                for sub in sub_contexts:
                    contexts.append(sub + [top_val])
            return contexts
    
    all_contexts = generate_all_contexts(n_levels)
    n_contexts = len(all_contexts)
    n_trials = n_contexts * trials_per_context
    
    # Create prototypes and outcomes for each context
    prototypes = {}
    outcomes = {}
    
    for ctx_id, context in enumerate(all_contexts):
        features = [0] * total_features
        
        # Encode top-level features
        for i in range(n_top_levels):
            level_idx = n_levels - i - 1
            features[i] = context[level_idx]
        
        # Encode species features (if applicable)
        if n_levels >= 2:
            species_val = context[1]
            features[n_top_levels] = species_val
            features[n_top_levels + 1] = species_val
        
        # Encode color features (if applicable)
        if n_levels >= 1:
            color_val = context[0]
            features[n_top_levels + n_species_feats + color_val] = 1
        
        # Determine outcome based on task rules
        if n_levels <= 1:
            outcome = 0
        elif n_levels == 2:
            outcome = 1 if context[1] == 0 else 0
        else:
            top_val = context[n_levels - 1]
            second_val = context[n_levels - 2]
            outcome = 1 if (top_val == 0 and second_val == 0) else 0
        
        prototypes[ctx_id] = features
        outcomes[ctx_id] = outcome
    
    # Generate all trials with small probability of noise
    all_trials = []
    for ctx_id in range(n_contexts):
        for _ in range(trials_per_context):
            features = prototypes[ctx_id].copy()
            
            # Add noise to color features with 2% probability
            if rng.random() < 0.02:
                color_feat_idx = rng.choice(range(n_top_levels + n_species_feats, 
                                                  n_top_levels + n_species_feats + n_color_feats))
                features[color_feat_idx] = 1 - features[color_feat_idx]
            
            all_trials.append({
                'context_id': ctx_id,
                'features': features,
                'outcome': outcomes[ctx_id],
                'context': all_contexts[ctx_id]
            })
    
    # Shuffle trials
    rng.shuffle(all_trials)
    
    # Create feature matrix and track true factors
    F = np.zeros((total_features + 1, n_trials), dtype=int)
    true_factors = {i: [] for i in range(1, n_levels + 1)}
    
    for t, trial in enumerate(all_trials):
        F[:total_features, t] = trial['features']
        F[outcome_idx, t] = trial['outcome']
        
        for level in range(1, n_levels + 1):
            true_factors[level].append(trial['context'][level - 1])
    
    # Create metadata dictionary
    true_factors_dict = {}
    for level in range(1, n_levels + 1):
        true_factors_dict[f'level_{level}'] = np.array(true_factors[level])
    
    # Add legacy keys for compatibility
    if n_levels >= 2:
        true_factors_dict['true_species'] = true_factors_dict['level_2']
    if n_levels >= 3:
        true_factors_dict['true_preparations'] = true_factors_dict['level_3']
    
    feedback_mask = np.ones((total_features + 1, n_trials), dtype=bool)
    
    meta = {
        'n_levels': n_levels,
        'n_contexts': n_contexts,
        'true_factors': true_factors_dict,
        'outcome_idx': outcome_idx,
        'n_features': total_features,
    }
    
    return F, meta, feedback_mask


# ============================================================================
# COMPRESSION METRICS
# ============================================================================

def compute_cluster_redundancy_metrics(
    particles,
    meta,
    task_type="species",
    min_valid_frac=0.10,
    normalize_entropy=True,
    weight_entropy_by_freq=True,
):
    """
    Measure representational redundancy and compression efficiency.
    
    Args:
        particles: Particle assignments (n_particles x n_trials)
        meta: Task metadata with true labels
        task_type: Which task level to analyze (or level number as string)
        min_valid_frac: Minimum fraction of valid (non -1) assignments
        normalize_entropy: Whether to normalize entropy to [0,1]
        weight_entropy_by_freq: Whether to weight by label frequency
    
    Returns:
        Dictionary with compression metrics, or None if insufficient data
    """
    from scipy.stats import entropy as scipy_entropy

    particles = np.asarray(particles)
    n_particles, n_trials = particles.shape

    # Get majority-vote cluster assignment for each trial
    cluster_assignments = np.full(n_trials, -1, dtype=int)
    for t in range(n_trials):
        vec = particles[:, t]
        vec = vec[vec >= 0]
        if vec.size:
            u, c = np.unique(vec, return_counts=True)
            cluster_assignments[t] = int(u[np.argmax(c)])

    # Get true labels based on task type
    if task_type == "species":
        if "true_species" in meta:
            true_labels = np.asarray(meta["true_species"], dtype=int)
        elif "true_groups" in meta:
            true_labels = np.asarray([meta["true_groups"][ctx] for ctx in meta["true_contexts"]], dtype=int)
        else:
            return None
    elif task_type == "preparation":
        if "true_preparations" in meta:
            true_labels = np.asarray(meta["true_preparations"], dtype=int)
        else:
            return None
    else:
        # For other levels, use the level key
        level_key = f'level_{task_type}'
        
        if 'true_factors' not in meta:
            return None
        if level_key not in meta['true_factors']:
            return None
        
        true_labels = np.asarray(meta['true_factors'][level_key], dtype=int)

    # Check coverage
    valid_mask = cluster_assignments >= 0
    valid_frac = float(np.mean(valid_mask))
    n_valid = int(np.sum(valid_mask))

    if valid_frac < min_valid_frac or n_valid < 5:
        return None

    # Filter to valid assignments only
    cluster_assignments = cluster_assignments[valid_mask]
    true_labels = true_labels[valid_mask]

    # Need at least 2 labels
    unique_labels = np.unique(true_labels)
    n_labels = len(unique_labels)
    if n_labels < 2:
        return None

    unique_clusters = np.unique(cluster_assignments)
    n_clusters = len(unique_clusters)

    # METRIC 1: Clusters per label
    clusters_per_label = {}
    for lab in unique_labels:
        mask = true_labels == lab
        clusters_per_label[int(lab)] = int(len(np.unique(cluster_assignments[mask])))
    avg_clusters_per_label = float(np.mean(list(clusters_per_label.values())))

    # METRIC 2: Redundancy ratio
    redundancy_ratio = float((n_clusters - n_labels) / n_labels) if n_labels > 0 else 0.0

    # METRIC 3: Entropy per label
    entropies = []
    entropies_norm = []
    weights = []

    for lab in unique_labels:
        mask = true_labels == lab
        clusters_for_lab = cluster_assignments[mask]
        u, c = np.unique(clusters_for_lab, return_counts=True)
        probs = c / np.sum(c)

        H = float(scipy_entropy(probs))
        entropies.append(H)

        if normalize_entropy:
            K = len(u)
            Hn = float(H / np.log(K)) if K > 1 else 0.0
            entropies_norm.append(Hn)

        weights.append(float(np.mean(mask)) if weight_entropy_by_freq else 1.0)

    weights = np.asarray(weights, float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    avg_entropy = float(np.sum(weights * np.asarray(entropies)))
    avg_entropy_norm = float(np.sum(weights * np.asarray(entropies_norm))) if normalize_entropy else np.nan

    # METRIC 4: Modal purity
    modal_purities = []
    for lab in unique_labels:
        mask = true_labels == lab
        clusters_for_lab = cluster_assignments[mask]
        _, c = np.unique(clusters_for_lab, return_counts=True)
        purity = float(np.max(c) / np.sum(c))
        modal_purities.append(purity)

    avg_modal_purity = float(np.sum(weights * np.asarray(modal_purities)))

    # METRIC 5: Parsimony score
    parsimony_score = float(n_labels / n_clusters) if n_clusters > 0 else 0.0

    return {
        "valid_frac": valid_frac,
        "n_valid": n_valid,
        "n_trials": int(n_trials),
        "n_clusters": int(n_clusters),
        "n_labels": int(n_labels),
        "avg_clusters_per_label": avg_clusters_per_label,
        "redundancy_ratio": redundancy_ratio,
        "parsimony_score": parsimony_score,
        "avg_entropy": avg_entropy,
        "avg_entropy_norm": avg_entropy_norm,
        "avg_modal_purity": avg_modal_purity,
    }

# ============================================================================
# Structure metrics
# ============================================================================

def compute_structural_alignment_metrics(particles, meta, task_level=2):
    """
    Compute ARI and NMI for structural alignment with ground truth.
    
    Args:
        particles: Particle assignments (n_particles x n_trials)
        meta: Task metadata with true factors
        task_level: Which level to evaluate (2=species, 3=prep, etc.)
    
    Returns:
        Dictionary with ARI and NMI, or None if insufficient data
    """
    particles = np.asarray(particles)
    n_particles, n_trials = particles.shape
    
    # Get majority-vote cluster assignment
    cluster_assignments = np.full(n_trials, -1, dtype=int)
    for t in range(n_trials):
        vec = particles[:, t]
        vec = vec[vec >= 0]
        if vec.size:
            u, c = np.unique(vec, return_counts=True)
            cluster_assignments[t] = int(u[np.argmax(c)])
    
    # Get ground truth labels
    level_key = f'level_{task_level}'
    if 'true_factors' not in meta or level_key not in meta['true_factors']:
        return None
    
    true_labels = np.asarray(meta['true_factors'][level_key], dtype=int)
    
    # Filter to valid assignments
    valid_mask = (cluster_assignments >= 0)
    n_valid = int(np.sum(valid_mask))
    
    if n_valid < 2:
        return None
    
    cluster_assignments = cluster_assignments[valid_mask]
    true_labels = true_labels[valid_mask]
    
    # Compute ARI and NMI
    ari = adjusted_rand_score(true_labels, cluster_assignments)
    nmi = normalized_mutual_info_score(true_labels, cluster_assignments)
    
    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'n_valid': n_valid,
        'n_clusters': len(np.unique(cluster_assignments)),
        'n_true_labels': len(np.unique(true_labels))
    }


def compute_hierarchical_structural_alignment(paths_hier, meta, task_level=2):
    """
    Compute ARI and NMI for hierarchical model at each tree level.
    
    Args:
        paths_hier: Particle paths (nParticles, nTrials, max_depth)
        meta: Task metadata
        task_level: Which ground truth level to compare against
    
    Returns:
        Dictionary with alignment metrics at each level
    """
    if paths_hier is None:
        return None
    
    nParticles, nTrials, max_depth = paths_hier.shape
    
    # Get ground truth labels
    level_key = f'level_{task_level}'
    if 'true_factors' not in meta or level_key not in meta['true_factors']:
        return None
    
    true_labels = np.asarray(meta['true_factors'][level_key], dtype=int)
    
    # Find deepest level used
    deepest_level = 0
    for L in range(max_depth):
        if np.any(paths_hier[:, :, L] >= 0):
            deepest_level = L
    
    # Compute alignment at each level
    ari_by_level = {}
    nmi_by_level = {}
    
    for L in range(deepest_level + 1):
        # Majority vote assignments at this level
        assignments = np.zeros(nTrials, dtype=int)
        for t in range(nTrials):
            nodes = paths_hier[:, t, L]
            valid = nodes[nodes >= 0]
            if len(valid) > 0:
                unique, counts = np.unique(valid, return_counts=True)
                assignments[t] = unique[np.argmax(counts)]
            else:
                assignments[t] = -1
        
        # Filter valid
        valid_mask = (assignments >= 0)
        
        if np.sum(valid_mask) >= 2:
            ari = adjusted_rand_score(true_labels[valid_mask], assignments[valid_mask])
            nmi = normalized_mutual_info_score(true_labels[valid_mask], assignments[valid_mask])
        else:
            ari = 0.0
            nmi = 0.0
        
        ari_by_level[f'level_{L}'] = float(ari)
        nmi_by_level[f'level_{L}'] = float(nmi)
    
    # Best alignment across all levels
    best_ari = max(ari_by_level.values()) if ari_by_level else 0.0
    best_nmi = max(nmi_by_level.values()) if nmi_by_level else 0.0
    
    # Which level achieved best ARI
    best_level = max(ari_by_level.items(), key=lambda x: x[1])[0] if ari_by_level else 'level_0'
    
    return {
        'ari': best_ari,
        'nmi': best_nmi,
        'best_level': best_level,
        'ari_by_level': ari_by_level,
        'nmi_by_level': nmi_by_level,
        'deepest_level': deepest_level
    }


def compute_outcome_accuracy_trace(rEst, true_outcomes):
    """
    Compute trial-by-trial outcome prediction accuracy.
    
    Args:
        rEst: Outcome estimates (n_trials,)
        true_outcomes: True outcomes (n_trials,)
    
    Returns:
        Array of trial-by-trial accuracy (0 or 1 per trial)
    """
    predicted = (rEst > 0.5).astype(int)
    accuracy_trace = (predicted == true_outcomes).astype(float)
    return accuracy_trace


# ============================================================================
# MAIN LOOP - WITH F1 METRICS
# ============================================================================

def run_single_seed_all_tasks(seed, max_levels, alpha, omega):
    """
    Run one seed across all task complexities 
    
    INCLUDES:
    1. Outcome prediction accuracy + TRACE
    2. Transfer accuracy with F1/precision/recall metrics
    3. Compression metrics
    4. STRUCTURAL ALIGNMENT (ARI/NMI) 
    5. Learning traces 
    """
    results = []
    
    if seed % 5 == 0:
        log_progress(f"Starting seed {seed}")
    
    for n_levels in range(2, max_levels + 1):
        # Generate task
        F, meta, fb = generate_scalable_hierarchical_task_quiet(
            n_levels=n_levels,
            trials_per_context=10,
            seed=seed
        )
        
        n_trials = F.shape[1]
        outcome_idx = meta['outcome_idx']
        true_outcomes = F[outcome_idx, :]
        
        # Train flat model
        np.random.seed(seed)
        p_flat, _, rEst_flat, _ = one_layer_inference_loop(
            nTrials=n_trials, nParticles=200, nFeatures=F.shape[0],
            regime=omega, alpha=alpha, f=F,
            outcome_idx_per_trial=np.full(n_trials, outcome_idx, dtype=int),
            feedback_mask=fb[outcome_idx, :], random_seed=seed,
        )
        
        # Train hierarchical model
        np.random.seed(seed)
        p_hier, _, rEst_hier, _, _, paths_hier, _ = full_hier_inference_loop(
            nTrials=n_trials, nParticles=200, nFeatures=F.shape[0],
            alpha=alpha, omega=omega, f=F, max_depth=20, max_children=20,
            outcome_idx=outcome_idx, feedback_mask=fb, random_seed=seed,
        )

        
        seed_result = {
            'seed': seed, 
            'n_levels': n_levels,
            'n_trials': n_trials,
            'n_contexts': meta['n_contexts']
        }
        
        # ====================================================================
        # 1. OUTCOME PREDICTION 
        # ====================================================================
        
        # Compute traces
        flat_outcome_trace = compute_outcome_accuracy_trace(rEst_flat, true_outcomes)
        hier_outcome_trace = compute_outcome_accuracy_trace(rEst_hier, true_outcomes)
        
        # Store traces
        seed_result['flat_outcome_trace'] = flat_outcome_trace.tolist()
        seed_result['hier_outcome_trace'] = hier_outcome_trace.tolist()
        
        # Final accuracy
        flat_outcome_acc = float(np.mean(flat_outcome_trace))
        hier_outcome_acc = float(np.mean(hier_outcome_trace))
        
        seed_result['flat_outcome_acc'] = flat_outcome_acc
        seed_result['hier_outcome_acc'] = hier_outcome_acc
        seed_result['outcome_advantage'] = hier_outcome_acc - flat_outcome_acc
        
        # ====================================================================
        # 2. STRUCTURAL ALIGNMENT (ARI/NMI) 
        # ====================================================================
        
        level_names = {2: 'species', 3: 'prep', 4: 'storage', 5: 'season'}
        
        for test_level in range(2, n_levels + 1):
            level_name = level_names.get(test_level, f'level_{test_level}')
            
            # FLAT MODEL ARI/NMI
            flat_alignment = compute_structural_alignment_metrics(
                particles=p_flat,
                meta=meta,
                task_level=test_level
            )
            
            if flat_alignment:
                seed_result[f'{level_name}_flat_ari'] = flat_alignment['ari']
                seed_result[f'{level_name}_flat_nmi'] = flat_alignment['nmi']
                seed_result[f'{level_name}_flat_ari_n_clusters'] = flat_alignment['n_clusters']
            
            # HIERARCHICAL MODEL ARI/NMI (across all levels)
            hier_alignment = compute_hierarchical_structural_alignment(
                paths_hier=paths_hier,
                meta=meta,
                task_level=test_level
            )
            
            if hier_alignment:
                seed_result[f'{level_name}_hier_ari'] = hier_alignment['ari']
                seed_result[f'{level_name}_hier_nmi'] = hier_alignment['nmi']
                seed_result[f'{level_name}_hier_ari_best_level'] = hier_alignment['best_level']
                seed_result[f'{level_name}_hier_ari_by_level'] = hier_alignment['ari_by_level']
                seed_result[f'{level_name}_hier_nmi_by_level'] = hier_alignment['nmi_by_level']
                seed_result[f'{level_name}_hier_tree_depth'] = hier_alignment['deepest_level']
        
        # ====================================================================
        # 3. COMPRESSION METRICS 
        # ====================================================================
        
        for test_level in range(2, n_levels + 1):
            level_name = level_names.get(test_level, f'level_{test_level}')
            
            flat_metrics = compute_cluster_redundancy_metrics(
                particles=p_flat,
                meta=meta,
                task_type=str(test_level),
                min_valid_frac=0.10,
            )
            
            if flat_metrics:
                prefix = f'{level_name}_flat'
                seed_result[f'{prefix}_n_clusters'] = flat_metrics['n_clusters']
                seed_result[f'{prefix}_n_labels'] = flat_metrics['n_labels']
                seed_result[f'{prefix}_redundancy_ratio'] = flat_metrics['redundancy_ratio']
                seed_result[f'{prefix}_parsimony_score'] = flat_metrics['parsimony_score']
                seed_result[f'{prefix}_avg_entropy'] = flat_metrics['avg_entropy']
                seed_result[f'{prefix}_avg_modal_purity'] = flat_metrics['avg_modal_purity']
                seed_result[f'{prefix}_valid_frac'] = flat_metrics['valid_frac']
        
        # ====================================================================
        # 4. TRANSFER ACCURACY WITH F1 METRICS
        # ====================================================================
        
        for test_true_level in range(2, n_levels + 1):
            name = level_names.get(test_true_level, f'level_{test_true_level}')
            level_key = f'level_{test_true_level}'
            true_factor = meta['true_factors'][level_key]
            
            factor_0_trials = np.where(true_factor == 0)[0]
            if len(factor_0_trials) == 0:
                continue
            
            labeled_trial = int(factor_0_trials[0])
            
            # ----------------------------------------------------------------
            # FLAT MODEL TRANSFER WITH F1 METRICS
            # ----------------------------------------------------------------
            flat_cluster = _majority_id(p_flat[:, labeled_trial])
            flat_pred = np.array([0 if _majority_id(p_flat[:, t]) == flat_cluster else 1 
                                 for t in range(n_trials)])
            
            # Compute all metrics
            flat_metrics = compute_transfer_metrics_with_f1(flat_pred, true_factor, positive_label=0)
            
            seed_result[f'{name}_flat_transfer'] = flat_metrics['accuracy']
            seed_result[f'{name}_flat_precision'] = flat_metrics['precision']
            seed_result[f'{name}_flat_recall'] = flat_metrics['recall']
            seed_result[f'{name}_flat_f1'] = flat_metrics['f1']
            seed_result[f'{name}_flat_TP'] = flat_metrics['TP']
            seed_result[f'{name}_flat_FP'] = flat_metrics['FP']
            seed_result[f'{name}_flat_FN'] = flat_metrics['FN']
            seed_result[f'{name}_flat_TN'] = flat_metrics['TN']
            
            # ----------------------------------------------------------------
            # HIERARCHICAL MODEL TRANSFER WITH F1 METRICS
            # ----------------------------------------------------------------
            paths_level = test_true_level - 1
            hier_metrics = None
            level_used = -1
            
            for try_level in range(paths_level, -1, -1):
                if try_level >= paths_hier.shape[2]:
                    continue
                
                nodes = np.array([_majority_id(paths_hier[:, t, try_level]) 
                                 for t in range(n_trials)])
                valid_frac = np.mean(nodes >= 0)
                
                if valid_frac >= 0.10:
                    hier_node = _majority_id(paths_hier[:, labeled_trial, try_level])
                    
                    if hier_node >= 0:
                        hier_pred = np.array([0 if _majority_id(paths_hier[:, t, try_level]) == hier_node 
                                             else 1 for t in range(n_trials)])
                        
                        # Compute all metrics
                        hier_metrics = compute_transfer_metrics_with_f1(hier_pred, true_factor, positive_label=0)
                        level_used = try_level
                        break
            
            # Store hierarchical metrics
            if hier_metrics:
                seed_result[f'{name}_hier_transfer'] = hier_metrics['accuracy']
                seed_result[f'{name}_hier_precision'] = hier_metrics['precision']
                seed_result[f'{name}_hier_recall'] = hier_metrics['recall']
                seed_result[f'{name}_hier_f1'] = hier_metrics['f1']
                seed_result[f'{name}_hier_TP'] = hier_metrics['TP']
                seed_result[f'{name}_hier_FP'] = hier_metrics['FP']
                seed_result[f'{name}_hier_FN'] = hier_metrics['FN']
                seed_result[f'{name}_hier_TN'] = hier_metrics['TN']
                seed_result[f'{name}_hier_level_used'] = level_used
                
                # Compute advantages
                seed_result[f'{name}_transfer_advantage'] = hier_metrics['accuracy'] - flat_metrics['accuracy']
                seed_result[f'{name}_f1_advantage'] = hier_metrics['f1'] - flat_metrics['f1']
                seed_result[f'{name}_precision_advantage'] = hier_metrics['precision'] - flat_metrics['precision']
                seed_result[f'{name}_recall_advantage'] = hier_metrics['recall'] - flat_metrics['recall']
            else:
                seed_result[f'{name}_hier_transfer'] = np.nan
                seed_result[f'{name}_hier_precision'] = np.nan
                seed_result[f'{name}_hier_recall'] = np.nan
                seed_result[f'{name}_hier_f1'] = np.nan
                seed_result[f'{name}_hier_level_used'] = -1
            
            # ----------------------------------------------------------------
            # HIERARCHICAL COMPRESSION (at same level as transfer)
            # ----------------------------------------------------------------
            if level_used >= 0 and level_used < paths_hier.shape[2]:
                hier_particles = paths_hier[:, :, int(level_used)]
                
                hier_comp_metrics = compute_cluster_redundancy_metrics(
                    particles=hier_particles,
                    meta=meta,
                    task_type=str(test_true_level),
                    min_valid_frac=0.10,
                )
                
                if hier_comp_metrics:
                    prefix = f'{name}_hier'
                    seed_result[f'{prefix}_n_clusters'] = hier_comp_metrics['n_clusters']
                    seed_result[f'{prefix}_n_labels'] = hier_comp_metrics['n_labels']
                    seed_result[f'{prefix}_redundancy_ratio'] = hier_comp_metrics['redundancy_ratio']
                    seed_result[f'{prefix}_parsimony_score'] = hier_comp_metrics['parsimony_score']
                    seed_result[f'{prefix}_avg_entropy'] = hier_comp_metrics['avg_entropy']
                    seed_result[f'{prefix}_avg_modal_purity'] = hier_comp_metrics['avg_modal_purity']
                    seed_result[f'{prefix}_valid_frac'] = hier_comp_metrics['valid_frac']
                    seed_result[f'{prefix}_compression_level'] = int(level_used)
        
        results.append(seed_result)
    
    return results


# ============================================================================
# PARALLEL ANALYSIS RUNNER
# ============================================================================

def run_parallel_analysis(max_levels=5, n_seeds=100, alpha=2.0, omega=0.5, n_workers=None):
    """
    Run analysis in parallel across seeds.
    
    Args:
        max_levels: Maximum task complexity to test
        n_seeds: Number of random seeds to run
        alpha: CRP concentration parameter
        omega: Stickiness parameter
        n_workers: Number of parallel workers
    
    Returns:
        Flat list of all results across seeds and task complexities
    """
    if n_workers is None:
        n_workers = min(16, max(1, cpu_count() - 2))
    
    log_progress(f"Starting parallel analysis with {n_workers} workers")
    log_progress(f"Parameters: alpha={alpha:.3f}, omega={omega:.3f}")
    log_progress(f"Total seeds: {n_seeds}, Tasks per seed: {max_levels-1}")
    
    start = time.time()
    
    # Create partial function with fixed parameters
    worker_func = partial(run_single_seed_all_tasks, 
                         max_levels=max_levels, 
                         alpha=alpha, 
                         omega=omega)
    
    # Run in parallel
    with Pool(n_workers) as pool:
        results_nested = pool.map(worker_func, range(n_seeds))
    
    elapsed = time.time() - start
    log_progress(f"Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Flatten nested results
    all_results = [item for sublist in results_nested for item in sublist]
    
    return all_results

# ============================================================================
# ANALYSIS HELPERS - WITH F1 SUPPORT
# ============================================================================

def compute_outcome_prediction_performance(results, n_levels=2):
    """
    Compute mean outcome prediction accuracy for a given task complexity.
    
    Args:
        results: List of result dictionaries
        n_levels: Task complexity level to analyze
    
    Returns:
        Tuple of (flat_mean, hier_mean, hier_advantage)
    """
    df = pd.DataFrame(results)
    df_n = df[df['n_levels'] == n_levels]
    
    if len(df_n) == 0:
        return 0.0, 0.0, 0.0
    
    flat_vals = df_n['flat_outcome_acc'].dropna().values
    hier_vals = df_n['hier_outcome_acc'].dropna().values
    
    flat_mean = np.mean(flat_vals) if len(flat_vals) > 0 else 0.0
    hier_mean = np.mean(hier_vals) if len(hier_vals) > 0 else 0.0
    advantage = hier_mean - flat_mean
    
    return flat_mean, hier_mean, advantage

def compute_outcome_prediction_performance_avg(results, levels=(2, 3, 4, 5)):
    """Average outcome prediction accuracy across multiple task complexity levels."""
    all_flat, all_hier = [], []
    for n in levels:
        flat, hier, _ = compute_outcome_prediction_performance(results, n_levels=n)
        if hier > 0.0:
            all_flat.append(flat)
            all_hier.append(hier)
    if not all_hier:
        return 0.0, 0.0, 0.0
    flat_mean = np.mean(all_flat)
    hier_mean = np.mean(all_hier)
    return flat_mean, hier_mean, hier_mean - flat_mean

def compute_transfer_performance(results, n_levels=None, metric='accuracy'):
    """
    Compute mean transfer performance (accuracy, F1, precision, or recall).
    
    Args:
        results: List of result dictionaries
        n_levels: Specific task level to evaluate (None = average across 3-5 level tasks)
        metric: Which metric to compute ('accuracy', 'f1', 'precision', 'recall')
    
    Returns:
        Tuple of (flat_mean, hier_mean, hier_advantage)
    """
    df = pd.DataFrame(results)
    level_names = {2: 'species', 3: 'prep', 4: 'storage', 5: 'season'}
    
    # Map metric to column suffix
    if metric == 'accuracy':
        suffix = 'transfer'
    else:
        suffix = metric
    
    if n_levels is not None:
        # Compute for specific level
        df_n = df[df['n_levels'] == n_levels]
        level_name = level_names.get(n_levels)
        
        if level_name and f'{level_name}_hier_{suffix}' in df_n.columns:
            hier_vals = df_n[f'{level_name}_hier_{suffix}'].dropna().values
            flat_vals = df_n[f'{level_name}_flat_{suffix}'].dropna().values
            
            if len(hier_vals) > 0:
                flat_mean = np.mean(flat_vals)
                hier_mean = np.mean(hier_vals)
                advantage = hier_mean - flat_mean
                return flat_mean, hier_mean, advantage
        
        return 0.0, 0.0, 0.0
    
    else:
        # Average across transfer tasks (3-5 levels)
        all_flat = []
        all_hier = []
        
        for test_n_levels in [3, 4, 5]:
            df_n = df[df['n_levels'] == test_n_levels]
            level_name = level_names.get(test_n_levels)
            
            if level_name and f'{level_name}_hier_{suffix}' in df_n.columns:
                hier_vals = df_n[f'{level_name}_hier_{suffix}'].dropna().values
                flat_vals = df_n[f'{level_name}_flat_{suffix}'].dropna().values
                
                if len(hier_vals) > 0:
                    all_flat.append(np.mean(flat_vals))
                    all_hier.append(np.mean(hier_vals))
        
        # Return averages across levels
        if len(all_hier) > 0:
            flat_mean = np.mean(all_flat)
            hier_mean = np.mean(all_hier)
            advantage = hier_mean - flat_mean
            return flat_mean, hier_mean, advantage
        else:
            return 0.0, 0.0, 0.0

# ============================================================================
# PARAMETER SWEEP - WITH F1 OPTIMIZATION OPTION
# ============================================================================

def print_comprehensive_summary(results):
    """
    Print comprehensive summary of all metrics including F1.
    
    Args:
        results: List of result dictionaries
    """
    df = pd.DataFrame(results)
    
    log_progress("Computing comprehensive summary...")
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    level_names = {2: 'species', 3: 'prep', 4: 'storage', 5: 'season'}
    
    for n in range(2, 6):
        df_n = df[df['n_levels'] == n]
        if len(df_n) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"{n}-LEVEL TASK")
        print(f"{'='*80}")
        
        # 1. OUTCOME PREDICTION
        flat_outcome = df_n['flat_outcome_acc'].dropna().values
        hier_outcome = df_n['hier_outcome_acc'].dropna().values
        
        if len(hier_outcome) > 0:
            print(f"\n1. OUTCOME PREDICTION:")
            print(f"   Flat:  {np.mean(flat_outcome):.1%} +/- {np.std(flat_outcome):.1%}")
            print(f"   Hier:  {np.mean(hier_outcome):.1%} +/- {np.std(hier_outcome):.1%}")
            print(f"   Adv:   {np.mean(hier_outcome)-np.mean(flat_outcome):+.1%}")
        
        # 2. TRANSFER PERFORMANCE WITH F1
        name = level_names.get(n)
        if name and f'{name}_hier_transfer' in df_n.columns:
            # Accuracy
            hier_acc = df_n[f'{name}_hier_transfer'].dropna().values
            flat_acc = df_n[f'{name}_flat_transfer'].dropna().values
            
            # F1
            hier_f1 = df_n[f'{name}_hier_f1'].dropna().values
            flat_f1 = df_n[f'{name}_flat_f1'].dropna().values
            
            # Precision & Recall
            hier_prec = df_n[f'{name}_hier_precision'].dropna().values
            flat_prec = df_n[f'{name}_flat_precision'].dropna().values
            hier_rec = df_n[f'{name}_hier_recall'].dropna().values
            flat_rec = df_n[f'{name}_flat_recall'].dropna().values
            
            if len(hier_acc) > 0:
                print(f"\n2. TRANSFER PERFORMANCE (at {name} level):")
                print(f"   Accuracy:  Flat={np.mean(flat_acc):.1%}, Hier={np.mean(hier_acc):.1%}, Adv={np.mean(hier_acc)-np.mean(flat_acc):+.1%}")
                print(f"   F1 Score:  Flat={np.mean(flat_f1):.3f}, Hier={np.mean(hier_f1):.3f}, Adv={np.mean(hier_f1)-np.mean(flat_f1):+.3f}")
                print(f"   Precision: Flat={np.mean(flat_prec):.3f}, Hier={np.mean(hier_prec):.3f}")
                print(f"   Recall:    Flat={np.mean(flat_rec):.3f}, Hier={np.mean(hier_rec):.3f}")
        
        # 3. COMPRESSION METRICS
        if name and f'{name}_flat_n_clusters' in df_n.columns:
            flat_clusters = df_n[f'{name}_flat_n_clusters'].dropna().values
            hier_clusters = df_n[f'{name}_hier_n_clusters'].dropna().values
            flat_entropy = df_n[f'{name}_flat_avg_entropy'].dropna().values
            hier_entropy = df_n[f'{name}_hier_avg_entropy'].dropna().values
            
            if len(hier_clusters) > 0:
                print(f"\n3. COMPRESSION (at {name} level):")
                print(f"   Clusters: Flat={np.mean(flat_clusters):.1f}, Hier={np.mean(hier_clusters):.1f}")
                print(f"   Entropy:  Flat={np.mean(flat_entropy):.3f}, Hier={np.mean(hier_entropy):.3f}")


def run_parameter_sweep(n_param_settings=10, n_seeds=10, 
                       alpha_range=(0.5, 5.0), omega_range=(0.1, 2.0),
                       optimization_objective='outcome_prediction',
                       optimize_on_level=2):
    """
    Run parameter sweep with flexible optimization objective including F1.
    
    Args:
        n_param_settings: Number of parameter combinations to test
        n_seeds: Number of seeds per parameter setting
        alpha_range: (min, max) for alpha sampling
        omega_range: (min, max) for omega sampling
        optimization_objective: What to optimize for:
            - 'outcome_prediction': Optimize for outcome prediction accuracy
            - 'transfer': Optimize for transfer accuracy
            - 'transfer_advantage': Optimize for hierarchical advantage in transfer
            - 'transfer_f1': Optimize for transfer F1 score
            - 'f1_advantage': Optimize for hierarchical advantage in F1
        optimize_on_level: Which task level to optimize on (default: 2)
    
    Returns:
        best_alpha: Optimal alpha value
        best_omega: Optimal omega value
        sweep_results: List of all results with parameters and outcomes
    """
    log_progress("="*80)
    log_progress("PARAMETER SWEEP")
    log_progress("="*80)
    log_progress(f"Testing {n_param_settings} parameter combinations with {n_seeds} seeds each")
    log_progress(f"Alpha range: {alpha_range}, Omega range: {omega_range}")
    log_progress(f"Optimization objective: {optimization_objective}")
    if optimization_objective == 'outcome_prediction':
        log_progress(f"Optimizing on: {optimize_on_level}-level task")
    elif optimize_on_level is not None:
        log_progress(f"Optimizing on: {optimize_on_level}-level transfer task")
    else:
        log_progress(f"Optimizing on: Average across 3-5 level transfer tasks")
    
    # Generate random parameter combinations
    rng = np.random.RandomState(13) 
    param_combos = []
    
    for i in range(n_param_settings):
        alpha = rng.uniform(alpha_range[0], alpha_range[1])
        omega = rng.uniform(omega_range[0], omega_range[1])
        param_combos.append((alpha, omega))
    
    # Test each combination
    sweep_results = []
    best_score = -np.inf
    best_params = None
    
    for idx, (alpha, omega) in enumerate(param_combos):
        log_progress(f"\n[{idx+1}/{n_param_settings}] Testing alpha={alpha:.3f}, omega={omega:.3f}")
        
        # Run analysis
        results = run_parallel_analysis(
            max_levels=5,
            n_seeds=n_seeds,
            alpha=alpha,
            omega=omega,
            n_workers=None
        )
        
        # Compute outcome prediction metrics
        if optimization_objective == 'outcome_prediction':
            opt_level = optimize_on_level if optimize_on_level is not None else 2
        else:
            opt_level = 2  # Always compute base task for reporting
        
        flat_outcome, hier_outcome, outcome_adv = compute_outcome_prediction_performance(
            results, n_levels=opt_level
        )
        
        # Compute transfer metrics (accuracy)
        if optimization_objective in ['transfer', 'transfer_advantage']:
            transfer_opt_level = optimize_on_level
        else:
            transfer_opt_level = None
        
        flat_transfer, hier_transfer, transfer_adv = compute_transfer_performance(
            results, n_levels=transfer_opt_level, metric='accuracy'
        )
        
        # Compute F1 metrics
        flat_f1, hier_f1, f1_adv = compute_transfer_performance(
            results, n_levels=transfer_opt_level, metric='f1'
        )
        
        # Store results
        sweep_results.append({
            'alpha': alpha,
            'omega': omega,
            'outcome_prediction_flat': flat_outcome,
            'outcome_prediction_hier': hier_outcome,
            'outcome_prediction_advantage': outcome_adv,
            'transfer_flat': flat_transfer,
            'transfer_hier': hier_transfer,
            'transfer_advantage': transfer_adv,
            'f1_flat': flat_f1,
            'f1_hier': hier_f1,
            'f1_advantage': f1_adv,
            'results': results
        })
        
        # Determine optimization score based on objective
        if optimization_objective == 'outcome_prediction':
            optimization_score = hier_outcome
            score_name = f"Outcome prediction (hier, {opt_level}-level)"
        elif optimization_objective == 'outcome_prediction_avg':
            flat_outcome, hier_outcome, outcome_adv = compute_outcome_prediction_performance_avg(
                results, levels=(2, 3, 4, 5))
            optimization_score = hier_outcome
            score_name = "Outcome prediction (hier, avg all levels)"
        elif optimization_objective == 'transfer':
            optimization_score = hier_transfer
            if transfer_opt_level is not None:
                score_name = f"Transfer accuracy (hier, {transfer_opt_level}-level)"
            else:
                score_name = "Transfer accuracy (hier, avg 3-5 level)"
        elif optimization_objective == 'transfer_advantage':
            optimization_score = transfer_adv
            if transfer_opt_level is not None:
                score_name = f"Transfer advantage ({transfer_opt_level}-level)"
            else:
                score_name = "Transfer advantage (avg 3-5 level)"
        elif optimization_objective == 'transfer_f1':
            optimization_score = hier_f1
            if transfer_opt_level is not None:
                score_name = f"Transfer F1 (hier, {transfer_opt_level}-level)"
            else:
                score_name = "Transfer F1 (hier, avg 3-5 level)"
        elif optimization_objective == 'f1_advantage':
            optimization_score = f1_adv
            if transfer_opt_level is not None:
                score_name = f"F1 advantage ({transfer_opt_level}-level)"
            else:
                score_name = "F1 advantage (avg 3-5 level)"
        else:
            raise ValueError(f"Unknown optimization_objective: {optimization_objective}")
        
        log_progress(f"  -> Outcome:  Flat={flat_outcome:.1%}, Hier={hier_outcome:.1%}, Adv={outcome_adv:+.1%}")
        log_progress(f"  -> Transfer: Flat={flat_transfer:.1%}, Hier={hier_transfer:.1%}, Adv={transfer_adv:+.1%}")
        log_progress(f"  -> F1:       Flat={flat_f1:.3f}, Hier={hier_f1:.3f}, Adv={f1_adv:+.3f}")
        log_progress(f"  -> Optimization score ({score_name}): {optimization_score:.3f}")
        
        # Track best parameters
        if optimization_score > best_score:
            best_score = optimization_score
            best_params = (alpha, omega)
            log_progress(f"  ✓ New best {score_name}!")
    
    log_progress("\n" + "="*80)
    log_progress("PARAMETER SWEEP COMPLETE")
    log_progress("="*80)
    log_progress(f"Best parameters: alpha={best_params[0]:.3f}, omega={best_params[1]:.3f}")
    log_progress(f"Best score ({score_name}): {best_score:.3f}")
    
    # Find all metrics for best params
    best_result = [r for r in sweep_results 
                   if r['alpha'] == best_params[0] and r['omega'] == best_params[1]][0]
    log_progress(f"With best params:")
    log_progress(f"  Outcome (hier):     {best_result['outcome_prediction_hier']:.1%}")
    log_progress(f"  Transfer (hier):    {best_result['transfer_hier']:.1%}")
    log_progress(f"  F1 (hier):          {best_result['f1_hier']:.3f}")
    log_progress(f"  Transfer advantage: {best_result['transfer_advantage']:+.1%}")
    log_progress(f"  F1 advantage:       {best_result['f1_advantage']:+.3f}")
    
    # Print top 5 results
    print("\n" + "="*80)
    print(f"TOP 5 PARAMETER COMBINATIONS (by {optimization_objective})")
    print("="*80)
    
    # Sort by optimization objective
    if optimization_objective in ('outcome_prediction', 'outcome_prediction_avg'):
        sorted_results = sorted(sweep_results,
                            key=lambda x: x['outcome_prediction_hier'],
                            reverse=True)
    elif optimization_objective == 'transfer':
        sorted_results = sorted(sweep_results, 
                               key=lambda x: x['transfer_hier'], 
                               reverse=True)
    elif optimization_objective == 'transfer_advantage':
        sorted_results = sorted(sweep_results, 
                               key=lambda x: x['transfer_advantage'], 
                               reverse=True)
    elif optimization_objective == 'transfer_f1':
        sorted_results = sorted(sweep_results,
                               key=lambda x: x['f1_hier'],
                               reverse=True)
    elif optimization_objective == 'f1_advantage':
        sorted_results = sorted(sweep_results,
                               key=lambda x: x['f1_advantage'],
                               reverse=True)
    
    for i, res in enumerate(sorted_results[:5]):
        print(f"{i+1}. alpha={res['alpha']:.3f}, omega={res['omega']:.3f}")
        print(f"   Outcome (hier):     {res['outcome_prediction_hier']:.1%}")
        print(f"   Transfer (hier):    {res['transfer_hier']:.1%}")
        print(f"   F1 (hier):          {res['f1_hier']:.3f}")
        print(f"   Transfer adv:       {res['transfer_advantage']:+.1%}")
        print(f"   F1 adv:             {res['f1_advantage']:+.3f}")
    
    # Save sweep results
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'results_{args.optimize}_{date_str}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(sweep_results, f)
    log_progress(f"Yay! Sweep results saved to {filename}")

    
    return best_params[0], best_params[1], sweep_results


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter optimization')
    parser.add_argument('--optimize', type=str, default='f1_advantage',
                    choices=['outcome_prediction', 'outcome_prediction_avg',
                                'transfer', 'transfer_advantage',
                                'transfer_f1', 'f1_advantage'],
                    help='What to optimize for')
    parser.add_argument('--level', type=int, default=None,
                       help='Level to optimize on (None = average for transfer)')
    parser.add_argument('--n_param_settings', type=int, default=200,
                       help='Number of parameter combinations to test')
    parser.add_argument('--n_seeds_sweep', type=int, default=6,
                       help='Number of seeds per parameter combo in sweep')

    
    args = parser.parse_args()
    
    log_progress("="*80)
    log_progress("COMPLETE ANALYSIS - WITH F1 METRICS")
    log_progress("="*80)
    log_progress(f"CPUs available: {cpu_count()}")
    log_progress(f"Optimization objective: {args.optimize}")
    
    # STEP 1: Parameter sweep
    log_progress("\n" + "="*80)
    log_progress("STEP 1: PARAMETER OPTIMIZATION")
    log_progress("="*80)
    
    # Determine optimization level
    if args.level is not None:
        optimize_level = args.level
    else:
        # Default levels based on objective
        if args.optimize == 'outcome_prediction':
            optimize_level = 2  # Base task
        else:
            optimize_level = None  # Average across transfer tasks
    
    best_alpha, best_omega, sweep_results = run_parameter_sweep(
        n_param_settings=args.n_param_settings,
        n_seeds=args.n_seeds_sweep,
        alpha_range=(0.1, 3.0),
        omega_range=(0.1, 3.0),
        optimization_objective=args.optimize,
        optimize_on_level=optimize_level
    )

    log_progress(f"\n✓ Optimal parameters: alpha={best_alpha:.3f}, omega={best_omega:.3f}")
    
    # Save everything
    final_output = {
        'best_alpha': best_alpha,
        'best_omega': best_omega,
        'optimization_objective': args.optimize,
        'optimization_level': optimize_level,
        'note': f'Parameters optimized for {args.optimize} with F1 metrics',
        'sweep_results': sweep_results,
        'final_results': [],
        'columns_explanation': {
            'outcome_acc': 'Accuracy on outcome prediction',
            'transfer': 'Accuracy on transfer test',
            'precision': 'Transfer precision (TP / (TP + FP))',
            'recall': 'Transfer recall (TP / (TP + FN))',
            'f1': 'Transfer F1 score (harmonic mean of precision and recall)',
            'TP': 'True positives (correctly identified same category)',
            'FP': 'False positives (incorrectly called same category - overgeneralization)',
            'FN': 'False negatives (incorrectly called different category - over-discrimination)',
            'TN': 'True negatives (correctly identified different category)',
            'compression': 'Representational efficiency metrics',
            'n_clusters': 'Number of distinct clusters learned',
            'parsimony_score': 'n_labels / n_clusters (1.0 = perfect)',
            'redundancy_ratio': '(n_clusters - n_labels) / n_labels (0 = perfect)',
            'avg_entropy': 'Entropy of cluster assignments per label (lower = better)'
        }
    }

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'results_{args.optimize}_{date_str}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(final_output, f)
    log_progress(f"Yippie! Complete results saved to {filename}")
    
    log_progress("\n" + "="*80)
    log_progress("ANALYSIS COMPLETE!")
    log_progress("="*80)
    log_progress(f"Optimal parameters: alpha={best_alpha:.3f}, omega={best_omega:.3f}")
    log_progress(f"Optimized for: {args.optimize}")
    log_progress("Evaluated: Outcome prediction, transfer (with F1/precision/recall), and compression")
    log_progress("All metrics saved for analysis!")