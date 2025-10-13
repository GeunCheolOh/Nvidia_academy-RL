import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(episodes, returns, success_rates, save_path=None):
    """
    Plot training metrics: episode returns and success rates.
    
    Args:
        episodes: List of episode numbers
        returns: List of episode returns
        success_rates: List of success rates
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot episode returns
    ax1.plot(episodes, returns)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns Over Time')
    ax1.grid(True)
    
    # Plot success rates
    ax2.plot(episodes, success_rates)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_moving_average(episodes, values, window_size=100, title="Moving Average", ylabel="Value", save_path=None):
    """
    Plot moving average of values.
    
    Args:
        episodes: List of episode numbers
        values: List of values to average
        window_size: Size of moving average window
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional path to save the plot
    """
    if len(values) < window_size:
        window_size = len(values)
    
    moving_avg = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(values[start_idx:i+1]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, moving_avg)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(f'{title} (Window Size: {window_size})')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_q_table_heatmap(q_table, map_size=(4, 4), save_path=None):
    """
    Plot Q-table as heatmap for FrozenLake environment.
    
    Args:
        q_table: Q-table array (states x actions)
        map_size: Tuple of (rows, cols) for the environment
        save_path: Optional path to save the plot
    """
    rows, cols = map_size
    
    # Get max Q-values for each state
    max_q_values = np.max(q_table, axis=1)
    
    # Reshape to grid
    q_grid = max_q_values.reshape(rows, cols)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(q_grid, cmap='viridis')
    plt.colorbar(label='Max Q-Value')
    plt.title('Q-Table Heatmap (Max Q-Values per State)')
    
    # Add text annotations
    for i in range(rows):
        for j in range(cols):
            state = i * cols + j
            text = f'{max_q_values[state]:.2f}'
            plt.text(j, i, text, ha='center', va='center', color='white', fontweight='bold')
    
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_policy_arrows(q_table, map_size=(4, 4), save_path=None):
    """
    Plot learned policy as arrows on the grid.
    
    Args:
        q_table: Q-table array (states x actions)
        map_size: Tuple of (rows, cols) for the environment
        save_path: Optional path to save the plot
    """
    rows, cols = map_size
    
    # Action mappings: 0=Left, 1=Down, 2=Right, 3=Up
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    # Get best action for each state
    best_actions = np.argmax(q_table, axis=1)
    
    plt.figure(figsize=(8, 6))
    
    # Create grid
    for i in range(rows):
        for j in range(cols):
            state = i * cols + j
            best_action = best_actions[state]
            arrow = action_arrows[best_action]
            
            plt.text(j, i, arrow, ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Draw grid lines
            plt.axhline(y=i-0.5, color='black', linewidth=1)
            plt.axvline(x=j-0.5, color='black', linewidth=1)
    
    plt.axhline(y=rows-0.5, color='black', linewidth=1)
    plt.axvline(x=cols-0.5, color='black', linewidth=1)
    
    plt.xlim(-0.5, cols-0.5)
    plt.ylim(-0.5, rows-0.5)
    plt.gca().invert_yaxis()
    
    plt.title('Learned Policy (Best Actions)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Remove ticks
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()