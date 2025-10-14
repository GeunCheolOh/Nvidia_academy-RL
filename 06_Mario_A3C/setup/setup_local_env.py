"""
Setup script for A2C Mario learning environment.
This script creates a virtual environment and installs all required dependencies
for running the Super Mario Bros reinforcement learning project.

Features:
- Cross-platform support (Windows, macOS, Linux)
- Automatic virtual environment creation
- Dependency installation from requirements.txt
- Environment validation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class EnvironmentSetup:
    """Manages the setup of the local development environment."""

    def __init__(self):
        """Initialize the environment setup."""
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.system = platform.system()

    def get_python_executable(self):
        """
        Get the appropriate Python executable based on the platform.

        Returns:
            str: Path to the Python executable in the virtual environment
        """
        if self.system == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")

    def get_pip_executable(self):
        """
        Get the appropriate pip executable based on the platform.

        Returns:
            str: Path to the pip executable in the virtual environment
        """
        if self.system == "Windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")

    def create_virtual_environment(self):
        """
        Create a Python virtual environment.

        This uses the built-in venv module to create an isolated Python environment.
        """
        print(f"Creating virtual environment at {self.venv_path}...")

        if self.venv_path.exists():
            print(f"Virtual environment already exists at {self.venv_path}")
            response = input("Do you want to recreate it? (y/n): ")
            if response.lower() != 'y':
                print("Using existing virtual environment.")
                return
            else:
                print("Removing existing virtual environment...")
                import shutil
                shutil.rmtree(self.venv_path)

        subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
        print(f"Virtual environment created successfully at {self.venv_path}")

    def upgrade_pip(self):
        """Upgrade pip to the latest version."""
        print("Upgrading pip...")
        pip_executable = self.get_pip_executable()
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        print("Pip upgraded successfully.")

    def install_dependencies(self):
        """
        Install all required dependencies from requirements.txt.

        This includes PyTorch, Gymnasium, and other necessary packages.
        """
        print(f"Installing dependencies from {self.requirements_file}...")

        if not self.requirements_file.exists():
            raise FileNotFoundError(f"Requirements file not found at {self.requirements_file}")

        pip_executable = self.get_pip_executable()
        subprocess.run(
            [pip_executable, "install", "-r", str(self.requirements_file)],
            check=True
        )
        print("Dependencies installed successfully.")

    def verify_installation(self):
        """
        Verify that key packages are installed correctly.

        Tests imports of critical packages to ensure they work properly.
        """
        print("\nVerifying installation...")
        python_executable = self.get_python_executable()

        test_imports = [
            ("torch", "PyTorch"),
            ("gymnasium", "Gymnasium"),
            ("gym_super_mario_bros", "Super Mario Bros Environment"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("matplotlib", "Matplotlib"),
            ("tqdm", "tqdm"),
        ]

        all_successful = True
        for module, name in test_imports:
            try:
                result = subprocess.run(
                    [python_executable, "-c", f"import {module}; print('{name} imported successfully')"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✓ {result.stdout.strip()}")
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to import {name}")
                all_successful = False

        if all_successful:
            print("\n✓ All packages verified successfully!")
        else:
            print("\n✗ Some packages failed verification. Please check the installation.")
            return False

        return True

    def print_activation_instructions(self):
        """Print instructions for activating the virtual environment."""
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("\nTo activate the virtual environment, run:")

        if self.system == "Windows":
            print(f"  {self.venv_path}\\Scripts\\activate")
        else:
            print(f"  source {self.venv_path}/bin/activate")

        print("\nAfter activation, you can run:")
        print("  - Manual play: python games/test_manual_play.py")
        print("  - Training: python training/train_a2c.py")
        print("  - Demo: python games/demo_trained_agent.py")
        print("="*60)

    def run_setup(self):
        """
        Execute the complete setup process.

        This orchestrates all setup steps in the correct order.
        """
        print("="*60)
        print("A2C Super Mario Bros - Environment Setup")
        print(f"Platform: {self.system}")
        print(f"Python: {sys.version}")
        print("="*60)

        try:
            self.create_virtual_environment()
            self.upgrade_pip()
            self.install_dependencies()

            if self.verify_installation():
                self.print_activation_instructions()
                return True
            else:
                print("\nSetup completed with warnings. Please review the verification results.")
                return False

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error during setup: {e}")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            return False


def main():
    """Main entry point for the setup script."""
    setup = EnvironmentSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
