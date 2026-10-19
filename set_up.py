"""
Create a Python environment with everything the interpolator needs.

Usage (from the repository folder):
    python set_up.py

Then activate it and run the code:
    source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
    python main.py --help

Note: saving plots also needs LaTeX installed on your computer.
"""
import subprocess
import sys
import venv

PACKAGES = ["numpy", "scipy", "matplotlib", "tqdm", "emcee", "odrpack"]

# 1. Create the environment in the folder .venv
venv.create(".venv", with_pip=True)

# 2. Install the packages with the environment's own Python
python = ".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python"
subprocess.check_call([python, "-m", "pip", "install", *PACKAGES])

print("\nDone! Activate the environment with:  source .venv/bin/activate")
