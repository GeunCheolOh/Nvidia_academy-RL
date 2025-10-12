"""
Main entry point for Snake Genetic Algorithm project
"""
import sys
import os

def main():
    print("Snake Genetic Algorithm Project")
    print("=" * 50)
    print()
    print("Available scripts:")
    print("  train.py  - Train Genetic Algorithm models")
    print("  play.py   - Play with trained models (GUI)")
    print("  notebooks/ - Educational Jupyter notebooks")
    print()
    print("Quick start:")
    print("  python train.py --generations 50 --population_size 30")
    print("  python play.py --model models/snake_ga.npz")
    print()
    print("For detailed usage, run:")
    print("  python train.py --help")
    print("  python play.py --help")
    print()
    print("Jupyter notebooks:")
    print("  jupyter notebook notebooks/")

if __name__ == "__main__":
    main()