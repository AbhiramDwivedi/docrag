#!/usr/bin/env python3
"""
Post-clone setup script for GitHub users.
Run this after cloning the repository.
"""
import sys
import shutil
from pathlib import Path

def setup_project():
    """Setup the project after cloning from GitHub."""
    print("🚀 DocQuest Post-Clone Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("config").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   cd into the cloned repository first")
        return False
    
    # Copy template config if config.yaml doesn't exist
    config_path = Path("config/config.yaml")
    template_path = Path("config/config.yaml.template")
    
    if not config_path.exists() and template_path.exists():
        print("📁 Creating config from template...")
        shutil.copy2(template_path, config_path)
        print(f"✅ Created {config_path}")
        print("📝 Please edit config/config.yaml to:")
        print("   - Set your document folder path in sync_root")
        print("   - Add your OpenAI API key")
    elif config_path.exists():
        print(f"✅ Config file already exists: {config_path}")
    else:
        print("⚠️  Warning: No config template found")
    
    # Create data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print(f"✅ Created {data_dir} directory")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("⚠️  Warning: Python 3.11+ recommended for best performance")
    
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Edit config/config.yaml with your settings")
    print("3. Index documents: python -m backend.src.ingestion.pipeline --mode full")
    print("4. Ask questions: python -m backend.src.cli.ask \"What files are available?\"")
    print("\n📖 See README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    setup_project()
