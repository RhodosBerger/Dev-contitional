import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'advanced_cnc_copilot')))

def check_frontend():
    base = "advanced_cnc_copilot/frontend"
    files = ["index.html", "style.css", "app.js"]
    missing = []
    print("Checking Frontend Assets...")
    for f in files:
        if os.path.exists(os.path.join(base, f)):
            print(f"✅ Found {f}")
        else:
            print(f"❌ Missing {f}")
            missing.append(f)
    return len(missing) == 0

def check_prompt_library():
    print("\nChecking Prompt Library...")
    try:
        from backend.core.prompt_library import PromptLibrary
        presets = PromptLibrary.get_presets()
        print(f"✅ Loaded {len(presets)} presets.")
        for p in presets:
            print(f"   - {p['title']}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PromptLibrary: {e}")
        return False

if __name__ == "__main__":
    fe_ok = check_frontend()
    pl_ok = check_prompt_library()
    
    if fe_ok and pl_ok:
        print("\n🎉 UI & Prompt System Verified!")
    else:
        print("\n⚠️ Verification Failed.")
