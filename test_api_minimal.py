"""
Minimal Standalone Gemini API Test Script.
Does not depend on any agents, database, or Streamlit UI.
Tests authentication, model discovery, and content generation.
Never prints raw API keys - only masked previews.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def mask_key(k: str) -> str:
    if not k or len(k) < 8:
        return "Not configured / too short"
    return f"{k[:4]}...{k[-4:]}"

def run_test():
    print("=" * 60)
    print("🔍 MINIMAL GEMINI API DIAGNOSTIC TEST")
    print("=" * 60)

    # 1. Inspect and load .env files with override=True
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent

    env_paths = [
        base_dir / ".env",
        root_dir / ".env"
    ]

    loaded_from = None
    for p in env_paths:
        if p.exists():
            load_dotenv(p, override=True)
            loaded_from = p
            print(f"📁 Loaded environment from: {p}")
            break

    if not loaded_from:
        print("❌ No .env file found in project directories!")
        return False

    raw_gemini_key = os.getenv("GEMINI_API_KEY", "")
    raw_google_key = os.getenv("GOOGLE_API_KEY", "")

    # Strip whitespace/newlines
    gemini_key = raw_gemini_key.strip()
    google_key = raw_google_key.strip()

    print(f"🔑 GEMINI_API_KEY: {mask_key(gemini_key)} (length: {len(gemini_key)})")
    print(f"🔑 GOOGLE_API_KEY: {mask_key(google_key)} (length: {len(google_key)})")

    # Resolve active key
    active_key = gemini_key or google_key
    if not active_key or active_key == "your_gemini_api_key_here":
        print("❌ Error: API key is empty or still placeholder ('your_gemini_api_key_here').")
        return False

    # Force synchronization in os.environ so google-genai SDK does not use a stale key
    os.environ["GEMINI_API_KEY"] = active_key
    os.environ["GOOGLE_API_KEY"] = active_key

    # 2. Test google-genai Client initialization
    try:
        from google import genai
        from google.genai import types
        print("✅ google-genai SDK imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import google-genai SDK: {e}")
        return False

    try:
        client = genai.Client(api_key=active_key, http_options=types.HttpOptions(timeout=15000))
        print("✅ genai.Client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize genai.Client: {type(e).__name__} - {e}")
        return False

    # 3. Test Model Listing
    print("\n🔍 Querying Gemini Models API (client.models.list())...")
    generation_models = []
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions or "generate_content" in str(actions).lower():
                clean_name = m.name.replace("models/", "")
                generation_models.append(clean_name)
        print(f"✅ Discovered {len(generation_models)} content generation models.")
        print(f"📋 First 5 available models: {generation_models[:5]}")
    except Exception as e:
        print(f"❌ models.list() failed: {type(e).__name__} - {e}")
        return False

    # 4. Test Content Generation with active model
    # Prefer gemini-3.5-flash or first available
    preferred_models = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
    test_model = None
    for pm in preferred_models:
        if pm in generation_models:
            test_model = pm
            break
    if not test_model and generation_models:
        test_model = generation_models[0]

    print(f"\n🚀 Testing Content Generation with model '{test_model}'...")
    try:
        resp = client.models.generate_content(
            model=test_model,
            contents="Say 'Authentication Successful!' in exactly two words.",
            config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0)
        )
        if resp and resp.text:
            print("=" * 60)
            print(f"🎉 SUCCESS! API Key is fully authenticated and working!")
            print(f"🤖 Model: {test_model}")
            print(f"💬 Response: {resp.text.strip()}")
            print("=" * 60)
            return True
        else:
            print("⚠️ API returned an empty response.")
            return False
    except Exception as e:
        print(f"❌ Content generation failed: {type(e).__name__} - {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
