#!/usr/bin/env python3
"""
Setup script for Meeting Agent development environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Meeting Agent development environment...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Check if Poetry is installed
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✅ Poetry is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry is not installed")
        print("Please install Poetry: curl -sSL https://install.python-poetry.org | python3 -")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("poetry install", "Installing dependencies"):
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("\n📝 Creating .env file from template...")
        try:
            with open(".env.example", "r") as src:
                content = src.read()
            with open(".env", "w") as dst:
                dst.write(content)
            print("✅ .env file created")
            print("⚠️  Please edit .env file and add your API keys")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
    else:
        print("✅ .env file already exists")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    print("✅ Logs directory created")
    
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run API server: poetry run python api/main.py")
    print("3. Run WebSocket server: poetry run python websocket/handler.py")
    print("4. Visit http://localhost:8000/docs for API documentation")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
