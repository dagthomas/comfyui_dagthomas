#!/usr/bin/env python3
"""
Utility script to fetch and update Groq models list from the API.

Usage:
    python utils/update_groq_models.py

Requirements:
    GROQ_API_KEY environment variable must be set.
"""

import os
import json
import requests
from pathlib import Path


def fetch_groq_models():
    """Fetch available models from Groq API."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable is not set")
        print("Please set it with: export GROQ_API_KEY=your_key_here")
        return None
    
    try:
        print("🔄 Fetching models from Groq API...")
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        models_data = response.json()
        models = [model['id'] for model in models_data.get('data', [])]
        
        if not models:
            print("⚠️ No models found in API response")
            return None
        
        print(f"✅ Successfully fetched {len(models)} models")
        return sorted(models)
        
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def update_models_file(text_models, vision_models):
    """Update the groq_models.json file with fetched models."""
    # Get the path to the data directory
    script_dir = Path(__file__).parent.parent
    models_file = script_dir / "data" / "groq_models.json"
    
    # Create backup of existing file
    if models_file.exists():
        backup_file = models_file.with_suffix('.json.bak')
        print(f"📋 Creating backup: {backup_file.name}")
        with open(models_file, 'r') as f:
            backup_data = f.read()
        with open(backup_file, 'w') as f:
            f.write(backup_data)
    
    # Write new models
    models_data = {
        "text_models": text_models,
        "vision_models": vision_models,
        "updated_at": None,
        "source": "groq_api",
        "note": "Edit this file to add/remove models. Text models are for text generation, vision models support image analysis."
    }
    
    # Add timestamp if datetime is available
    try:
        from datetime import datetime
        models_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    except:
        pass
    
    print(f"💾 Updating {models_file.name}...")
    with open(models_file, 'w') as f:
        json.dump(models_data, f, indent=4)
    
    print(f"✅ Successfully updated {models_file.name}")
    return True


def display_models(all_models):
    """Display fetched models in a formatted way."""
    print("\n" + "="*60)
    print("📋 AVAILABLE GROQ MODELS")
    print("="*60)
    
    # Categorize models
    text_models = [m for m in all_models if not any(x in m.lower() for x in ['whisper', 'tts', 'guard'])]
    vision_models = [m for m in all_models if 'vision' in m.lower()]
    other_models = [m for m in all_models if m not in text_models and m not in vision_models]
    
    if text_models:
        print(f"\n🔤 Text/Chat Models ({len(text_models)}):")
        for model in text_models:
            print(f"  • {model}")
    
    if vision_models:
        print(f"\n👁️  Vision Models ({len(vision_models)}):")
        for model in vision_models:
            print(f"  • {model}")
    
    if other_models:
        print(f"\n🎙️  Other Models ({len(other_models)}) - Whisper/TTS/Guards:")
        for model in other_models:
            print(f"  • {model}")
    
    print("\n" + "="*60)
    return text_models, vision_models


def main():
    print("🚀 Groq Models Updater")
    print("="*60)
    
    # Fetch models from API
    all_models = fetch_groq_models()
    
    if not all_models:
        print("\n❌ Failed to fetch models from API")
        return 1
    
    # Display and categorize the models
    text_models, vision_models = display_models(all_models)
    
    # Ask for confirmation
    print("\n❓ Update groq_models.json with these models?")
    response = input("   Type 'yes' to continue: ").strip().lower()
    
    if response != 'yes':
        print("❌ Update cancelled")
        return 0
    
    # Update the file
    if update_models_file(text_models, vision_models):
        print("\n✅ All done! Restart ComfyUI to use the updated models.")
        return 0
    else:
        print("\n❌ Failed to update models file")
        return 1


if __name__ == "__main__":
    exit(main())

