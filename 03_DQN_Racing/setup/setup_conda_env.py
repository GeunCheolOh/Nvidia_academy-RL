#!/usr/bin/env python3
"""
DQN Racing Conda Environment Setup Script

This script sets up the DQN Racing environment using conda for cross-platform compatibility.
Conda automatically handles OS-specific dependencies including Box2D and SWIG.

Usage:
    python setup_conda_env.py
    
Requirements:
    - Anaconda or Miniconda installed
    - Internet connection

Author: DQN Racing Tutorial
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class CondaEnvironmentSetup:
    """Handles conda environment setup for DQN Racing project."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_name = "dqn_racing_conda"
        self.env_file = self.project_root / "environment.yml"
        self.system = platform.system().lower()
        
        # Detect conda command
        self.conda_cmd = self._detect_conda()
        
    def _detect_conda(self):
        """Detect available conda command."""
        commands = ['conda', 'mamba']  # mamba is faster alternative
        
        for cmd in commands:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ Found {cmd}: {result.stdout.strip()}")
                    return cmd
            except FileNotFoundError:
                continue
                
        return None
        
    def check_conda_installation(self):
        """Check if conda is installed and accessible."""
        if self.conda_cmd is None:
            print("❌ Conda not found!")
            print()
            print("Please install Anaconda or Miniconda:")
            print("📥 Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html")
            print("📥 Anaconda (full): https://www.anaconda.com/products/distribution")
            print()
            if self.system == "darwin":
                print("macOS quick install:")
                print("  brew install miniconda")
            elif self.system == "linux":
                print("Linux quick install:")
                print("  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
                print("  bash Miniconda3-latest-Linux-x86_64.sh")
            elif self.system == "windows":
                print("Windows: Download installer from website")
            return False
            
        return True
        
    def check_environment_file(self):
        """Check if environment.yml exists."""
        if not self.env_file.exists():
            print(f"❌ Environment file not found: {self.env_file}")
            return False
            
        print(f"✓ Environment file found: {self.env_file}")
        return True
        
    def remove_existing_environment(self):
        """Remove existing environment if it exists."""
        try:
            # Check if environment exists
            result = subprocess.run([
                self.conda_cmd, "env", "list"
            ], capture_output=True, text=True)
            
            if self.env_name in result.stdout:
                print(f"🗑️  Removing existing environment: {self.env_name}")
                subprocess.run([
                    self.conda_cmd, "env", "remove", "-n", self.env_name, "-y"
                ], check=True)
                print("✓ Existing environment removed")
            else:
                print(f"ℹ️  No existing environment found: {self.env_name}")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not remove existing environment: {e}")
            
    def create_environment(self):
        """Create conda environment from environment.yml."""
        print(f"🔧 Creating conda environment: {self.env_name}")
        print("This may take 5-10 minutes depending on your internet connection...")
        
        try:
            # Create environment from file
            result = subprocess.run([
                self.conda_cmd, "env", "create", "-f", str(self.env_file)
            ], check=True, capture_output=True, text=True)
            
            print("✓ Conda environment created successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create environment: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
            
    def verify_installation(self):
        """Verify that the environment was created correctly."""
        print("🔍 Verifying installation...")
        
        # Test script to run in the conda environment
        test_script = f'''
import sys
print(f"Python executable: {{sys.executable}}")

# Test imports
try:
    import torch
    print(f"✓ PyTorch {{torch.__version__}}")
    print(f"  CUDA available: {{torch.cuda.is_available()}}")
    
    import gymnasium as gym
    print(f"✓ Gymnasium {{gym.__version__}}")
    
    import pygame
    print(f"✓ Pygame {{pygame.version.ver}}")
    
    import numpy as np
    print(f"✓ NumPy {{np.__version__}}")
    
    import matplotlib
    print(f"✓ Matplotlib {{matplotlib.__version__}}")
    
    import cv2
    print(f"✓ OpenCV {{cv2.__version__}}")
    
    # Test CarRacing environment
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    obs, info = env.reset()
    print(f"✓ CarRacing environment: observation shape {{obs.shape}}")
    env.close()
    
    print("\\n🎉 All packages verified successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Verification error: {{e}}")
    sys.exit(1)
'''
        
        # Write test script to temporary file
        test_file = self.project_root / "temp_conda_test.py"
        test_file.write_text(test_script)
        
        try:
            # Run test script in conda environment
            result = subprocess.run([
                self.conda_cmd, "run", "-n", self.env_name, "python", str(test_file)
            ], check=True, capture_output=True, text=True)
            
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Verification failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
                
    def create_activation_scripts(self):
        """Create convenience scripts for activating the conda environment."""
        
        # Cross-platform activation script
        if self.system == "windows":
            script_name = "activate_conda_env.bat"
            script_content = f"""@echo off
echo Activating DQN Racing conda environment...
call conda activate {self.env_name}
echo Environment activated! 🚀
echo.
echo Available commands:
echo   python games/test_manual_play.py       - Test manual gameplay
echo   python tutorials/dqn_tutorial.py       - Run DQN tutorial  
echo   python training/dqn_training.py        - Start training
echo   python games/demo_trained_agent.py     - Demo trained agent
echo.
echo To deactivate: conda deactivate
cmd /k
"""
        else:
            script_name = "activate_conda_env.sh"
            script_content = f"""#!/bin/bash
echo "Activating DQN Racing conda environment..."
conda activate {self.env_name}
echo "Environment activated! 🚀"
echo ""
echo "Available commands:"
echo "  python games/test_manual_play.py       - Test manual gameplay"
echo "  python tutorials/dqn_tutorial.py       - Run DQN tutorial"
echo "  python training/dqn_training.py        - Start training"
echo "  python games/demo_trained_agent.py     - Demo trained agent"
echo ""
echo "To deactivate: conda deactivate"
exec "$SHELL"
"""
        
        script_path = self.project_root / script_name
        script_path.write_text(script_content)
        
        if self.system != "windows":
            # Make script executable on Unix systems
            os.chmod(script_path, 0o755)
            
        print(f"✓ Created activation script: {script_name}")
        
        # Also create environment info file
        info_content = f"""# DQN Racing Conda Environment

## Quick Start
```bash
# Activate environment
conda activate {self.env_name}

# Or use convenience script
./{script_name}   # Unix/macOS
{script_name}     # Windows
```

## Manual Commands
```bash
# Activate environment
conda activate {self.env_name}

# Test CarRacing game
python games/test_manual_play.py

# Run DQN tutorial
python tutorials/dqn_tutorial.py

# Start training
python training/dqn_training.py

# Test trained agent
python games/demo_trained_agent.py

# Deactivate environment
conda deactivate
```

## Environment Details
- Name: {self.env_name}
- Python: 3.8+
- PyTorch: 2.0+
- Box2D: ✓ (conda-forge)
- CarRacing: ✓ Ready to use

## Troubleshooting
If CarRacing doesn't work:
```bash
conda activate {self.env_name}
conda install -c conda-forge box2d-py --force-reinstall
```
"""
        
        readme_path = self.project_root / "CONDA_SETUP.md"
        readme_path.write_text(info_content)
        print(f"✓ Created setup guide: CONDA_SETUP.md")
        
    def setup(self):
        """Run complete conda environment setup."""
        print("=" * 60)
        print("🐍 DQN Racing Conda Environment Setup")
        print("=" * 60)
        
        try:
            # Check prerequisites
            if not self.check_conda_installation():
                return False
                
            if not self.check_environment_file():
                return False
                
            # Setup environment
            self.remove_existing_environment()
            
            if not self.create_environment():
                return False
                
            if not self.verify_installation():
                return False
                
            self.create_activation_scripts()
            
            print("\n" + "=" * 60)
            print("🎉 Conda setup completed successfully!")
            print("=" * 60)
            
            activation_cmd = f"conda activate {self.env_name}"
            if self.system == "windows":
                script_cmd = "activate_conda_env.bat"
            else:
                script_cmd = "./activate_conda_env.sh"
                
            print(f"🚀 To activate environment:")
            print(f"   {activation_cmd}")
            print(f"   OR")
            print(f"   {script_cmd}")
            print()
            print("📖 See CONDA_SETUP.md for detailed instructions")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False


def main():
    """Main function to run conda environment setup."""
    setup = CondaEnvironmentSetup()
    success = setup.setup()
    
    if not success:
        print("\n💡 Alternative options:")
        print("1. Use pip setup: python setup/setup_local_env.py")
        print("2. Install conda and try again")
        print("3. Use Docker (see agent.md)")
        sys.exit(1)


if __name__ == "__main__":
    main()