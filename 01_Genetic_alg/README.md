# Snake Genetic Algorithm

**Learn Genetic Algorithms through Snake game with real Python scripts and educational Jupyter notebooks**

This project provides a complete implementation of Genetic Algorithms for evolving neural networks to play the Snake game, featuring both practical Python scripts and educational materials.

## Features

### Complete Training & Inference Pipeline
- **Real Python Scripts**: Actual training and inference scripts you can run
- **GUI Visualization**: Watch your evolved snakes play with Pygame GUI
- **Snake Game Implementation**: Custom Snake game with proper visualization
- **Model Persistence**: Save and load evolved neural networks

### Educational Materials
- **5 Jupyter Notebooks**: Step-by-step learning progression
- **English Documentation**: No font issues, clear explanations
- **Concept-focused**: Theory with practical implementation

### Genetic Algorithm Implementation
- **Neural Network Evolution**: Evolve weights and biases
- **Multiple Selection Methods**: Tournament, Roulette, Rank selection
- **Crossover & Mutation**: Proper genetic operators
- **Elite Preservation**: Keep best individuals across generations

## Installation

### For Students

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd 1_1_Genetic_alg

# 2. Run automated setup (recommended)
chmod +x setup_env.sh
./setup_env.sh

# Or manual setup:
# python3 -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate
# pip install -r requirements.txt
```

### For Instructors (First Setup)

```bash
# 1. Navigate to project directory
cd 1_1_Genetic_alg

# 2. Initialize git (if not already done)
git init

# 3. Add all files
git add .

# 4. Create initial commit
git commit -m "Initial commit: Snake Genetic Algorithm educational project"

# 5. Add remote repository
git remote add origin <your-github-repo-url>

# 6. Push to GitHub
git push -u origin main
```

## Quick Start

### 1. Train a Model

```bash
# Train Genetic Algorithm
python train.py --generations 100 --population_size 50 --visualize

# Advanced training with custom parameters
python train.py --generations 200 --population_size 100 --mutation_rate 0.15 --visualize
```

### 2. Watch Your Evolved Snakes Play

```bash
# Play with trained model (opens GUI window)
python play.py --model models/snake_ga.npz --episodes 5 --show_stats

# Watch decisions in real-time
python play.py --model models/snake_ga_best.npz --episodes 3 --show_decisions
```

### 3. Learn with Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 01_snake_game_basics.ipynb
# 02_genetic_algorithm_basics.ipynb
# 03_snake_genetic_integration.ipynb
# 04_advanced_techniques.ipynb
# 05_real_world_applications.ipynb
```

## Project Structure

```
1_1_Genetic_alg/
├── train.py              # Main training script
├── play.py               # GUI inference script
├── main.py               # Project overview
├── setup_env.sh          # Environment setup script
├── core/                 # Core classes and utilities
│   └── base.py          # Abstract base classes
├── environments/         # Snake game environment
│   └── snake_env.py     # Pygame Snake implementation
├── algorithms/           # Genetic Algorithm implementation
│   └── genetic.py       # GA with neural networks
├── utils/                # Visualization and statistics
│   └── visualization.py # Evolution plots and analysis
├── notebooks/            # Educational Jupyter notebooks
│   ├── 01_snake_game_basics.ipynb
│   ├── 02_genetic_algorithm_basics.ipynb
│   ├── 03_snake_genetic_integration.ipynb
│   ├── 04_advanced_techniques.ipynb
│   └── 05_real_world_applications.ipynb
├── models/               # Saved models directory
└── results/              # Training results and plots
```

## Training Options

### Basic Training
```bash
python train.py --generations 50 --population_size 30
```

### Advanced Training
```bash
python train.py \
    --generations 200 \
    --population_size 100 \
    --mutation_rate 0.12 \
    --crossover_rate 0.8 \
    --selection_method tournament \
    --elite_size 10 \
    --visualize
```

### Custom Neural Network
```bash
python train.py \
    --hidden_layers 256 128 64 \
    --activation relu \
    --generations 100
```

## Inference Options

### Basic Playback
```bash
python play.py --model models/snake_ga.npz --episodes 3
```

### Detailed Analysis
```bash
python play.py \
    --model models/snake_ga_best.npz \
    --episodes 10 \
    --show_decisions \
    --show_stats \
    --speed 0.05
```

### Text Mode (No GUI)
```bash
python play.py --model models/snake_ga.npz --episodes 5 --text_mode
```

## Performance Monitoring

The training script automatically:
- Saves models periodically and keeps the best performing one
- Creates evolution plots if `--visualize` is used
- Logs progress every 5 generations
- Saves configuration and results

Training outputs include:
- `models/snake_ga.npz` - Latest model
- `models/snake_ga_best.npz` - Best performing model
- `results/evolution_plots.png` - Evolution curves
- `results/config.txt` - Training configuration

## Algorithm Details

### Genetic Algorithm
- **Representation**: Neural network weights as genome
- **Selection**: Tournament, Roulette wheel, or Rank-based
- **Crossover**: Uniform crossover between parent networks
- **Mutation**: Gaussian noise addition to weights
- **Elite Preservation**: Keep top performers each generation

### Neural Network
- **Architecture**: Fully connected feedforward
- **Input**: Flattened game state (board position, snake, food)
- **Output**: Action probabilities (Up, Right, Down, Left)
- **Activation**: ReLU, Tanh, or Sigmoid

### Fitness Function
- **Primary**: Food collection (100 points per food)
- **Secondary**: Survival time (0.1 points per step)
- **Penalty**: Early termination (-10 points)

## Learning Path

Follow the notebooks in order:

1. **Snake Game Basics** - Understand the environment and rules
2. **Genetic Algorithm Theory** - Learn GA fundamentals
3. **Snake-GA Integration** - Combine game with evolution
4. **Advanced Techniques** - Selection methods and operators
5. **Real-world Applications** - Extensions and improvements

Each notebook is self-contained with:
- Theory explanations
- Code implementations
- Interactive examples
- Visualization demonstrations

## Experimentation

### Compare Selection Methods
```bash
# Tournament selection
python train.py --selection_method tournament --tournament_size 5 --generations 50

# Roulette wheel selection
python train.py --selection_method roulette --generations 50

# Rank-based selection
python train.py --selection_method rank --generations 50
```

### Hyperparameter Tuning
```bash
# High mutation rate
python train.py --mutation_rate 0.2 --generations 50

# Large population
python train.py --population_size 200 --generations 30

# Deep networks
python train.py --hidden_layers 512 256 128 64 --generations 100
```

## Troubleshooting

### Common Issues

**1. Pygame installation problems:**
```bash
pip install pygame
# If that fails on macOS:
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
pip install pygame
```

**2. GUI not showing (headless systems):**
```bash
# Use text mode
python play.py --model models/snake_ga.npz --text_mode
```

**3. Memory issues with large populations:**
```bash
# Reduce population size
python train.py --population_size 30 --generations 100
```

## Expected Performance

- **Initial Generation**: Random behavior, scores 0-2
- **Generation 20-50**: Basic food seeking, scores 3-8
- **Generation 50-100**: Efficient play, scores 10-20+
- **Training time**: 30-60 minutes for 100 generations on modern hardware

Good evolved snakes typically:
- Avoid walls and self-collision
- Seek food efficiently
- Handle growing body length
- Score 15+ food items consistently

## Contributing

This is an educational project. Feel free to:
- Experiment with different fitness functions
- Try other neural network architectures
- Add new selection/crossover methods
- Implement other games

## License

This project is for educational purposes. Feel free to use and modify.

---

**Happy Learning!**
