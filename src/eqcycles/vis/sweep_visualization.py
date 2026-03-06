import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
from typing import Optional



def plot_pareto_front(sweep_df: pd.DataFrame):
    """
    A scatter plot of inversion_magnitude (X) vs. mass_recovery_pct (Y).
    Color: seq_weight (using LogNorm)
    Size: reg_m
    
    Args:
        sweep_df: DataFrame containing sweep results.
    """
    if sweep_df.empty:
        print("Warning: Sweep DataFrame is empty.")
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Scale sizes for reg_m for better visualization
    min_reg = sweep_df['reg_m'].min()
    max_reg = sweep_df['reg_m'].max()
    if max_reg > min_reg:
        sizes = (sweep_df['reg_m'] - min_reg) / (max_reg - min_reg) * 150 + 40
    else:
        sizes = [80] * len(sweep_df)
    
    # Use LogNorm for color if seq_weight has positive values
    has_pos_seq = (sweep_df['seq_weight'] > 0).any()
    norm = LogNorm() if has_pos_seq else None
    
    scatter = plt.scatter(
        sweep_df['inversion_magnitude'],
        sweep_df['mass_recovery_pct'],
        c=sweep_df['seq_weight'],
        s=sizes,
        cmap='viridis',
        norm=norm,
        alpha=0.8,
        edgecolors='w',
        linewidth=0.5
    )
    
    # 1. Colorbar for seq_weight
    plt.colorbar(scatter, label='Sequence Weight (seq_weight)')
    
    # 2. Custom Legend for reg_m (Size)
    if max_reg > min_reg:
        # Pick 4 representative values between the min and max reg_m
        legend_vals = np.linspace(min_reg, max_reg, num=4)
        # Calculate what their sizes would be using your exact formula
        legend_sizes = (legend_vals - min_reg) / (max_reg - min_reg) * 150 + 40
        
        # Create empty scatter points (proxy artists) to act as legend handles
        legend_handles = []
        for val, size in zip(legend_vals, legend_sizes):
            legend_handles.append(
                plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                            edgecolors='w', label=f'{val:.1f}')
            )
            
        # Add the size legend to the plot
        plt.legend(handles=legend_handles, title='reg_m size', 
                   loc='lower right', scatterpoints=1, framealpha=0.9)

    plt.xlabel('Inversion Magnitude (Topological Violations)')
    plt.ylabel('Mass Recovery (%)')
    plt.title('Hyperparameter Pareto Front: Mass Recovery vs. Sequence Integrity')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    return plt.gca()

def plot_parameter_heatmap(sweep_df: pd.DataFrame):
    """
    Plots a 3-panel heatmap showing 'best_score', 'mass_recovery_pct', 
    and 'inversion_magnitude' across the reg_m and seq_weight space.
    
    Args:
        sweep_df: DataFrame containing sweep results.
    """
    if sweep_df.empty:
        print("Warning: Sweep DataFrame is empty.")
        return None
        
    metrics = ['best_score', 'mass_recovery_pct', 'inversion_magnitude']
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    for ax, metric in zip(axes, metrics):
        # Pivot the data for seaborn heatmap
        pivot_table = sweep_df.pivot(index='reg_m', columns='seq_weight', values=metric)
        
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt=".2f", 
            cmap='YlGnBu' if metric != 'inversion_magnitude' else 'YlOrRd', 
            cbar_kws={'label': metric},
            ax=ax
        )
        
        ax.set_title(f'Sweep: {metric}')
        ax.set_xlabel('Sequence Weight (seq_weight)')
        ax.set_ylabel('Marginal Relaxation (reg_m)')
    
    plt.tight_layout()
    return axes


def plot_ensemble_pareto_front(aggregated_df: pd.DataFrame):
    """
    A scatter plot of mean inversion_magnitude (X) vs. mean mass_recovery_pct (Y)
    with error bars showing standard deviation.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_ensemble_metrics.
    """
    if aggregated_df.empty:
        print("Warning: Aggregated DataFrame is empty.")
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Scale sizes for reg_m
    min_reg = aggregated_df['reg_m'].min()
    max_reg = aggregated_df['reg_m'].max()
    if max_reg > min_reg:
        sizes = (aggregated_df['reg_m'] - min_reg) / (max_reg - min_reg) * 150 + 40
    else:
        sizes = [80] * len(aggregated_df)
    
    # Use LogNorm for color if seq_weight has positive values
    has_pos_seq = (aggregated_df['seq_weight'] > 0).any()
    norm = LogNorm() if has_pos_seq else None
    
    # Plot error bars first
    plt.errorbar(
        aggregated_df['inversion_magnitude_mean'],
        aggregated_df['mass_recovery_pct_mean'],
        xerr=aggregated_df['inversion_magnitude_std'],
        yerr=aggregated_df['mass_recovery_pct_std'],
        fmt='none',
        ecolor='gray',
        alpha=0.3,
        zorder=1
    )
    
    scatter = plt.scatter(
        aggregated_df['inversion_magnitude_mean'],
        aggregated_df['mass_recovery_pct_mean'],
        c=aggregated_df['seq_weight'],
        s=sizes,
        cmap='viridis',
        norm=norm,
        alpha=0.8,
        edgecolors='w',
        linewidth=0.5,
        zorder=2
    )
    
    plt.colorbar(scatter, label='Sequence Weight (seq_weight)')
    
    # Size legend
    if max_reg > min_reg:
        legend_vals = np.linspace(min_reg, max_reg, num=4)
        legend_sizes = (legend_vals - min_reg) / (max_reg - min_reg) * 150 + 40
        legend_handles = []
        for val, size in zip(legend_vals, legend_sizes):
            legend_handles.append(
                plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                            edgecolors='w', label=f'{val:.1f}')
            )
        plt.legend(handles=legend_handles, title='reg_m size', 
                   loc='lower right', scatterpoints=1, framealpha=0.9)

    plt.xlabel('Mean Inversion Magnitude (Topological Violations)')
    plt.ylabel('Mean Mass Recovery (%)')
    plt.title('Ensemble Hyperparameter Pareto Front (with Std Dev)')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    return plt.gca()


def plot_ensemble_robustness_heatmap(
    aggregated_df: pd.DataFrame,
    mass_recovery_threshold: float = 90.0,
    inversion_magnitude_threshold: float = 1.0
):
    """
    Plots a 4-panel heatmap showing ensemble metrics and stability across 
    the reg_m and seq_weight space.
    
    Metrics: success_rate, best_score, mass_recovery_pct, inversion_magnitude.
    Annotations include mean ± std for physical metrics to show robustness.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_ensemble_metrics.
        mass_recovery_threshold: Threshold for mass recovery used in success_rate.
        inversion_magnitude_threshold: Threshold for inversion magnitude used in success_rate.
    """
    if aggregated_df.empty:
        print("Warning: Aggregated DataFrame is empty.")
        return None
        
    metrics = [
        ('success_rate', None),
        ('best_score', 'best_score_std'),
        ('mass_recovery_pct', 'mass_recovery_pct_std'),
        ('inversion_magnitude', 'inversion_magnitude_std')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for ax, (metric_base, std_col) in zip(axes, metrics):
        mean_col = f"{metric_base}_mean" if metric_base != 'success_rate' else 'success_rate'
        
        # Pivot data for heatmap
        pivot_mean = aggregated_df.pivot(index='reg_m', columns='seq_weight', values=mean_col)
        
        # Prepare custom annotations: "mean \n ±std"
        if std_col:
            pivot_std = aggregated_df.pivot(index='reg_m', columns='seq_weight', values=std_col)
            annot_data = np.array([
                [f"{m:.2f}\n±{s:.2f}" for m, s in zip(row_m, row_s)]
                for row_m, row_s in zip(pivot_mean.values, pivot_std.values)
            ])
            label = metric_base
        else:
            # success_rate doesn't have a calculated std in the aggregation step
            annot_data = pivot_mean.round(2).astype(str).values
            label = f"Success Rate (Mass > {mass_recovery_threshold}%, Inv < {inversion_magnitude_threshold})"

        # Select colormap based on metric type
        if metric_base == 'success_rate':
            cmap = 'RdYlGn'
        elif 'inversion' in metric_base or 'score' in metric_base:
            # Lower is better for these
            cmap = 'YlOrRd'
        else:
            # Higher is better
            cmap = 'YlGnBu'
               
        sns.heatmap(
            pivot_mean, 
            annot=annot_data, 
            fmt="", 
            cmap=cmap, 
            cbar_kws={'label': label},
            ax=ax,
            annot_kws={"size": 9}
        )
        
        ax.set_title(f'Ensemble: {metric_base}')
        ax.set_xlabel('Sequence Weight (seq_weight)')
        ax.set_ylabel('Marginal Relaxation (reg_m)')
    
    plt.tight_layout()
    return axes
