import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ====================================================================================
# PLOTTING HELPER
# ====================================================================================
COLOR_FLAT = "#4D4B4B"
COLOR_HIER = '#7C3AED'
def create_bar_plot(data_flat, data_hier, title, ylabel, optimal_line=None, line_label='Optimal',
                   color_flat=COLOR_FLAT, color_hier=COLOR_HIER, show_points=True, 
                   use_ci=True, show_ci_shade=False, ci_alpha=0.15, show_stats = False):
    """
    Create a bar plot comparing flat vs hierarchical model performance.
    
    Args:
        data_flat: Array of flat model values
        data_hier: Array of hierarchical model values
        title: Plot title
        ylabel: Y-axis label
        optimal_line: If not None, draw horizontal reference line at this value, can also be the chance line
        line_label: Label for optimal line
        color_flat: Color for flat model bars
        color_hier: Color for hierarchical model bars
        show_points: If True, show individual data points
        use_ci: If True, use 95% CI; if False, use SEM
        show_ci_shade: If True, add shaded CI regions behind bars
        ci_alpha: Transparency of CI shading (default: 0.15)
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Compute statistics
    flat_mean = np.mean(data_flat)
    hier_mean = np.mean(data_hier)
    flat_sem = stats.sem(data_flat)
    hier_sem = stats.sem(data_hier)
    
    # Compute error bars based on use_ci
    if use_ci:
        flat_err = 1.96 * flat_sem
        hier_err = 1.96 * hier_sem
        error_type = '95% CI'
    else:
        flat_err = flat_sem
        hier_err = hier_sem
        error_type = 'SEM'
    
    data = {
        'Model': ['Flat', 'HOLMES'], 
        'Mean': [flat_mean, hier_mean], 
        'Error': [flat_err, hier_err]
    }
    df = pd.DataFrame(data)
    
    # Add shaded CI regions if requested (behind bars)
    if show_ci_shade:
        # Flat shaded region
        flat_rect = plt.Rectangle(
            (-0.4, flat_mean - flat_err),
            0.8,
            2 * flat_err,
            facecolor=color_flat,
            alpha=ci_alpha,
            edgecolor='none',
            zorder=1
        )
        ax.add_patch(flat_rect)
        
        # Hier shaded region
        hier_rect = plt.Rectangle(
            (0.6, hier_mean - hier_err),
            0.8,
            2 * hier_err,
            facecolor=color_hier,
            alpha=ci_alpha,
            edgecolor='black',
            zorder=1
        )
        ax.add_patch(hier_rect)
    
    # Create bar plot
    bars = ax.bar(df['Model'], df['Mean'], yerr=df['Error'],
                  color=[color_flat, color_hier], capsize=6,
                  edgecolor='black', linewidth=2, alpha=0.8,
                  error_kw={'linewidth': 2}, zorder=5)
    
    # Add individual points if enabled
    if show_points:
        np.random.seed(42)
        x_flat = np.random.normal(0, 0.05, len(data_flat))
        ax.scatter(x_flat, data_flat, color=color_flat, s=30, alpha=0.3,
                  edgecolors='white', linewidth=0.5, zorder=10)
        
        x_hier = np.random.normal(1, 0.05, len(data_hier))
        ax.scatter(x_hier, data_hier, color=color_hier, s=30, alpha=0.3,
                  edgecolors='white', linewidth=0.5, zorder=10)
    
    # # Add value labels on bars
    # for i, (bar, mean) in enumerate(zip(bars, df['Mean'])):
    #     height = bar.get_height()
    #     if height > 0.2:
    #         ax.text(bar.get_x() + bar.get_width() / 2., height - 0.1, 
    #                f'{height:.2f}', 
    #                ha='center', va='center', color='white', 
    #                fontsize=25, fontweight='bold')
    #     else:
    #         ax.text(bar.get_x() + bar.get_width() / 2., height + 0.2, 
    #                f'{height:.2f}', 
    #                ha='center', va='bottom', 
    #                fontsize=25, fontweight='bold')
    
    # Statistical test
    t_stat, p_val = stats.ttest_rel(data_hier, data_flat)
    print(f"\nStatistics for '{title}':")
    print(f"  Flat:  {flat_mean:.2f} ± {flat_err:.2f} ({error_type})")
    print(f"  Hier:  {hier_mean:.2f} ± {hier_err:.2f} ({error_type})")
    print(f"  Diff:  {hier_mean - flat_mean:+.2f}")
    print(f"  t({len(data_flat)-1}) = {t_stat:.3f}, p = {p_val:.2e}")
    
    if show_stats: 
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        y_max = max(df['Mean'] + df['Error']) + 0.3
        
        # Draw significance bracket
        ax.plot([0, 0, 1, 1], [y_max, y_max + 0.1, y_max + 0.1, y_max],
                'k-', linewidth=2, clip_on=False, zorder=15)
        
        # Add significance text
        if sig != 'n.s.':
            ax.text(0.5, y_max + 0.15, sig, ha='center', va='bottom',
                    fontsize=33, fontweight='bold', clip_on=False, zorder=15)
        else:
            ax.text(0.5, y_max + 0.15, sig, ha='center', va='bottom',
                    fontsize=20, fontweight='bold', clip_on=False, zorder=15)
    
    # Add optimal reference line if specified
    if optimal_line is not None:
        ax.axhline(optimal_line, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=line_label, zorder=2)
    
    # Labels and formatting
    ax.set_title('', fontsize=40, fontweight='bold', pad=40)
    ax.set_ylabel(ylabel, fontsize=28, fontweight='bold')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.grid(alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_elements = []
    if optimal_line is not None:
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                         linewidth=3, label=line_label))
    
    if show_ci_shade:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color_flat, 
                                            alpha=ci_alpha, label=f'{error_type} (shaded)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=20, loc='best')
    
    # # Add note about error bars
    # error_note = f'Error bars: {error_type}'
    # if show_ci_shade:
    #     error_note += f'\nShaded regions: {error_type}'
    # ax.text(0.98, 0.02, error_note,
    #         transform=ax.transAxes, fontsize=11,
    #         verticalalignment='bottom', horizontalalignment='right',
    #         style='italic',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()

