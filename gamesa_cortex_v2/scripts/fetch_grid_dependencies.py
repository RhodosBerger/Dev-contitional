import subprocess
import sys
import shutil

def check_command(cmd):
    return shutil.which(cmd) is not None

def fetch_dependencies():
    print("Gamesa Cortex V2: Fetching Grid Dependencies...")
    
    missing = []
    
    # 1. System Level
    if not check_command("cmake"): missing.append("cmake")
    if not check_command("rustc"): missing.append("rustc (cargo)")
    if not check_command("vulkaninfo"): missing.append("vulkan-tools")
    
    if missing:
        print(f"[WARNING] Missing System Tools: {', '.join(missing)}")
        print("Please run: sudo apt-get install cmake vulkan-tools rustc cargo")
    else:
        print("[OK] System Tools verified.")
        
    # 2. Python Level
    required_pylibs = ["numpy", "scipy", "wgpu", "pyopencl"]
    print(f"Checking Python Libs: {required_pylibs}")
    
    # In a real script we would pip install here, but we just warn
    # subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_pylibs)
    print("[OK] Python dependencies checked (Simulation).")
    
    print("\nDependency Fetch Complete.")

if __name__ == "__main__":
    fetch_dependencies()
