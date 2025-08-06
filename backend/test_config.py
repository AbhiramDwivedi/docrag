#!/usr/bin/env python3
"""Test configuration validation like CI does."""

import yaml
from pathlib import Path

print("=== Configuration Validation Test ===")

try:
    config = yaml.safe_load(Path('shared/config.yaml.template').read_text())
    required_keys = ['sync_root', 'db_path', 'vector_path', 'embed_model', 'openai_api_key']
    
    for key in required_keys:
        assert key in config, f'Missing required config key: {key}'
    
    print('✅ Configuration template is valid')
    print(f'✅ Found all required keys: {required_keys}')
    
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')

print("=== Done ===")
