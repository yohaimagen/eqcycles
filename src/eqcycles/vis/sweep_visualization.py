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


def plot_ensemble_robustness_heatmap(aggregated_df: pd.DataFrame):
    """
    Plots a heatmap of success_rate across the reg_m and seq_weight space.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_ensemble_metrics.
    """
    if aggregated_df.empty:
        print("Warning: Aggregated DataFrame is empty.")
        return None
        
    plt.figure(figsize=(10, 8))
    
    # Pivot the data
    pivot_table = aggregated_df.pivot(index='reg_m', columns='seq_weight', values='success_rate')
    
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".2f", 
        cmap='RdYlGn', 
        cbar_kws={'label': 'Success Rate (Mass > 80%, Inversion < 1.0)'}
    )
    
    plt.title('Ensemble Robustness Heatmap')
    plt.xlabel('Sequence Weight (seq_weight)')
    plt.ylabel('Marginal Relaxation (reg_m)')
    
    return plt.gca()
