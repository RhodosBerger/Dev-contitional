import os

files = [
    "README.md",
    "CONTRIBUTING.md",
    "requirements.txt",
    ".env.example",
    "docker-compose.yml"
]

missing = []
print("Verifying Configuration Files...")
for f in files:
    if os.path.exists(f):
        print(f"✅ Found {f}")
    else:
        print(f"❌ Missing {f}")
        missing.append(f)

if not missing:
    print("\n🎉 All Community Files Present!")
else:
    print("\n⚠️ Verification Failed.")
