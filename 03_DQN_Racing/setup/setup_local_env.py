#!/usr/bin/env python3
"""
DQN Racing Environment Setup Script

This script automatically sets up the development environment for DQN Racing learning.
It creates a virtual environment, installs dependencies, and verifies the installation.

Usage:
    python setup_local_env.py
"""

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path


class EnvironmentSetup:
    """Handles environment setup for DQN Racing project."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_name = "dqn_racing_env"
        self.venv_path = self.project_root / self.venv_name
        self.requirements_file = self.project_root / "requirements.txt"
        self.system = platform.system().lower()
        
    def detect_os(self):
        """Detect operating system and set appropriate commands."""
        print(f"Detected OS: {platform.system()} {platform.release()}")
        
        if self.system == "windows":
            self.python_cmd = "python"
            self.pip_cmd = str(self.venv_path / "Scripts" / "pip")
            self.activate_cmd = str(self.venv_path / "Scripts" / "activate")
        else:  # macOS, Linux, WSL
            self.python_cmd = "python3"
            self.pip_cmd = str(self.venv_path / "bin" / "pip")
            self.activate_cmd = f"source {self.venv_path / 'bin' / 'activate'}"
            
    def check_python_version(self):
        """Check if Python version is 3.8 or higher."""
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError("Python 3.8 or higher is required. Please upgrade Python.")
        print("✓ Python version check passed")
        
    def create_virtual_environment(self):
        """Create Python virtual environment."""
        if self.venv_path.exists():
            print(f"Virtual environment already exists at {self.venv_path}")
            print("Using existing virtual environment")
            return
                
        print(f"Creating virtual environment at {self.venv_path}")
        venv.create(self.venv_path, with_pip=True)
        print("✓ Virtual environment created successfully")
        
    def install_dependencies(self):
        """Install required packages from requirements.txt."""
        if not self.requirements_file.exists():
            raise FileNotFoundError(f"requirements.txt not found at {self.requirements_file}")
            
        print("Installing dependencies...")
        try:
            # Upgrade pip first
            subprocess.run([
                self.pip_cmd, "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            
            # Install core requirements first
            subprocess.run([
                self.pip_cmd, "install", "-r", str(self.requirements_file)
            ], check=True, capture_output=True, text=True)
            
            print("✓ Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
            
    def verify_installation(self):
        """Verify that key packages are installed correctly."""
        print("Verifying installation...")
        
        # Test script to run in the virtual environment
        test_script = '''
import sys
print(f"Python executable: {sys.executable}")

# Test imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    import gymnasium as gym
    print(f"✓ Gymnasium {gym.__version__}")
    
    import pygame
    print(f"✓ Pygame {pygame.version.ver}")
    
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
    
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
    
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
    
    # Test a simple environment first
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    print(f"✓ Basic Gymnasium working: observation shape {obs.shape}")
    env.close()
    
    # Test CarRacing environment (optional)
    try:
        env = gym.make('CarRacing-v3', render_mode='rgb_array')
        obs, info = env.reset()
        print(f"✓ CarRacing environment: observation shape {obs.shape}")
        env.close()
    except Exception as e:
        print(f"⚠️  CarRacing-v3 not available: {str(e)[:100]}...")
        print("   Will use alternative environment for now")
        print("   To enable CarRacing: pip install 'gymnasium[box2d]'")
    
    print("\\n✓ All packages verified successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Verification error: {e}")
    sys.exit(1)
'''
        
        # Write test script to temporary file
        test_file = self.project_root / "temp_test.py"
        test_file.write_text(test_script)
        
        try:
            # Run test script in virtual environment
            if self.system == "windows":
                python_executable = self.venv_path / "Scripts" / "python.exe"
            else:
                python_executable = self.venv_path / "bin" / "python"
                
            result = subprocess.run([
                str(python_executable), str(test_file)
            ], check=True, capture_output=True, text=True)
            
            print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Verification failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
                
    def create_activation_script(self):
        """Create convenience script for activating the environment."""
        if self.system == "windows":
            script_name = "activate_env.bat"
            script_content = f"""@echo off
echo Activating DQN Racing environment...
call "{self.activate_cmd}"
echo Environment activated! You can now run the training scripts.
echo.
echo Available commands:
echo   python games/test_manual_play.py       - Test manual gameplay
echo   python tutorials/dqn_tutorial.py       - Run DQN tutorial
echo   python training/dqn_training.py        - Start training
echo   python games/demo_trained_agent.py     - Demo trained agent
cmd /k
"""
        else:
            script_name = "activate_env.sh"
            script_content = f"""#!/bin/bash
echo "Activating DQN Racing environment..."
{self.activate_cmd}
echo "Environment activated! You can now run the training scripts."
echo ""
echo "Available commands:"
echo "  python games/test_manual_play.py       - Test manual gameplay"
echo "  python tutorials/dqn_tutorial.py       - Run DQN tutorial"
echo "  python training/dqn_training.py        - Start training"
echo "  python games/demo_trained_agent.py     - Demo trained agent"
exec "$SHELL"
"""
        
        script_path = self.project_root / script_name
        script_path.write_text(script_content)
        
        if self.system != "windows":
            # Make script executable on Unix systems
            os.chmod(script_path, 0o755)
            
        print(f"✓ Created activation script: {script_name}")
        
    def setup(self):
        """Run complete environment setup."""
        print("=" * 60)
        print("DQN Racing Environment Setup")
        print("=" * 60)
        
        try:
            self.detect_os()
            self.check_python_version()
            self.create_virtual_environment()
            self.install_dependencies()
            self.verify_installation()
            self.create_activation_script()
            
            print("\n" + "=" * 60)
            print("✓ Setup completed successfully!")
            print("=" * 60)
            
            if self.system == "windows":
                print("To activate the environment, run: activate_env.bat")
            else:
                print("To activate the environment, run: ./activate_env.sh")
                print("Or manually: source dqn_racing_env/bin/activate")
                
        except Exception as e:
            print(f"\n✗ Setup failed: {e}")
            print("\nPlease check the error messages above and try again.")
            sys.exit(1)


if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.setup()