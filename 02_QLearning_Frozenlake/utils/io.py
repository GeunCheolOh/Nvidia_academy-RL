import numpy as np
import os
import json
from typing import Dict, Any, Tuple


def save_q_table(q_table: np.ndarray, filepath: str) -> None:
    """
    Save Q-table to numpy file.
    
    Args:
        q_table: Q-table array to save
        filepath: Path to save the Q-table
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, q_table)
    print(f"Q-table saved to {filepath}")


def load_q_table(filepath: str) -> np.ndarray:
    """
    Load Q-table from numpy file.
    
    Args:
        filepath: Path to the Q-table file
        
    Returns:
        Loaded Q-table array
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Q-table file not found: {filepath}")
    
    q_table = np.load(filepath)
    print(f"Q-table loaded from {filepath}")
    print(f"Q-table shape: {q_table.shape}")
    return q_table


def save_training_log(log_data: Dict[str, Any], filepath: str) -> None:
    """
    Save training log data to JSON file.
    
    Args:
        log_data: Dictionary containing training metrics
        filepath: Path to save the log file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in log_data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        else:
            serializable_data[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Training log saved to {filepath}")


def load_training_log(filepath: str) -> Dict[str, Any]:
    """
    Load training log data from JSON file.
    
    Args:
        filepath: Path to the log file
        
    Returns:
        Dictionary containing training metrics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training log file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        log_data = json.load(f)
    
    print(f"Training log loaded from {filepath}")
    return log_data


def save_hyperparameters(hyperparams: Dict[str, Any], filepath: str) -> None:
    """
    Save hyperparameters to JSON file.
    
    Args:
        hyperparams: Dictionary containing hyperparameters
        filepath: Path to save the hyperparameters
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"Hyperparameters saved to {filepath}")


def load_hyperparameters(filepath: str) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON file.
    
    Args:
        filepath: Path to the hyperparameters file
        
    Returns:
        Dictionary containing hyperparameters
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    
    print(f"Hyperparameters loaded from {filepath}")
    return hyperparams


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def get_file_size(filepath: str) -> Tuple[int, str]:
    """
    Get file size in bytes and human-readable format.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Tuple of (size_in_bytes, human_readable_size)
    """
    if not os.path.exists(filepath):
        return 0, "0 B"
    
    size_bytes = os.path.getsize(filepath)
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return size_bytes, f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return size_bytes * (1024**4), f"{size_bytes:.1f} TB"