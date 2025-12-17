import shutil
import os
import site

# Target paths based on the error logs
site_packages = r"C:\Users\pc\AppData\Local\Programs\Python\Python313\Lib\site-packages"
targets = [
    os.path.join(site_packages, "matplotlib"),
    os.path.join(site_packages, "~atplotlib"),
    os.path.join(site_packages, "matplotlib-3.10.0.dist-info"),
    os.path.join(site_packages, "matplotlib-3.10.7.dist-info")
]

print(f"Cleaning targets in: {site_packages}")

for target in targets:
    if os.path.exists(target):
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
            print(f"✅ Removed: {target}")
        except Exception as e:
            print(f"❌ Failed to remove {target}: {e}")
    else:
        print(f"⚠️ Not found: {target}")

print("Cleanup complete.")
