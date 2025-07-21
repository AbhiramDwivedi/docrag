#!/usr/bin/env python3
"""Setup script to securely configure OpenAI API key."""
import sys
from pathlib import Path
import getpass
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

CONFIG_PATH = Path("config/config.yaml")

def setup_openai_key():
    """Securely prompt for and save OpenAI API key."""
    print("🔑 OpenAI API Key Setup")
    print("=" * 40)
    print("To use the question-answering feature, you need an OpenAI API key.")
    print("Get your key from: https://platform.openai.com/api-keys")
    print()
    
    # Load current config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"❌ Config file not found: {CONFIG_PATH}")
        return
    
    # Check if key is already set
    current_key = config.get('openai_api_key')
    if current_key and current_key != "your-openai-api-key-here" and current_key != "null":
        print(f"✅ API key is already configured (ends with: ...{current_key[-4:]})")
        update = input("Do you want to update it? (y/N): ").lower().strip()
        if update != 'y':
            print("No changes made.")
            return
    
    # Prompt for new key
    print("Please enter your OpenAI API key (input will be hidden):")
    api_key = getpass.getpass("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key entered. Setup cancelled.")
        return
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: OpenAI API keys usually start with 'sk-'")
        confirm = input("Continue anyway? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Setup cancelled.")
            return
    
    # Update config
    config['openai_api_key'] = api_key
    
    # Save config
    try:
        with open(CONFIG_PATH, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ API key saved to {CONFIG_PATH}")
        print("You can now use the CLI to ask questions about your documents!")
        print("\nExample: python -m cli.ask \"What Excel files are available?\"")
    except Exception as e:
        print(f"❌ Error saving config: {e}")

if __name__ == "__main__":
    setup_openai_key()
