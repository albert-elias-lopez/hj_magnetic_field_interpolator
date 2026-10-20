"""
Create a Python environment with everything the interpolator needs.

Usage (works from any folder):
    python set_up.py             install the versions used in the paper
                                 (requirements.txt, needs Python 3.11 or 3.12)
    python set_up.py --latest    install the newest versions instead
                                 (any Python >= 3.10; the last digits of the
                                 results may differ slightly from the paper)

The environment is created in the folder .venv of the repository. Running the
script again reuses it; delete .venv to start from scratch.

Then activate it and run the code:
    source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
    python main.py --help

LaTeX is optional: if it is installed, it is used for the plot labels.
"""
import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

PACKAGES = ["numpy", "scipy", "matplotlib", "tqdm", "emcee", "odrpack"]

repo = Path(__file__).resolve().parent
env = repo / ".venv"
python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

parser = argparse.ArgumentParser(description="Create the Python environment for the interpolator.")
parser.add_argument("--latest", action="store_true",
                    help="install the newest package versions instead of the ones used in the paper")
latest = parser.parse_args().latest

# 1. Check the Python version
version = sys.version_info[:2]
if version < (3, 10):
    sys.exit(f"Python {version[0]}.{version[1]} is too old: the code needs Python 3.10 or newer.")
if not latest and version not in [(3, 11), (3, 12)]:
    sys.exit(f"The package versions of the paper need Python 3.11 or 3.12, but this is Python "
             f"{version[0]}.{version[1]}.\nRun this script with Python 3.11 or 3.12 "
             f"(e.g. python3.12 set_up.py), or use  python set_up.py --latest  for the newest versions.")

# 2. Create the environment in the folder .venv (or reuse it)
if python.exists():
    print(f"Using the existing environment in {env}")
else:
    print(f"Creating the environment in {env}")
    try:
        venv.create(env, with_pip=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(env, ignore_errors=True)
        sys.exit(f"Could not create the environment (see the error above). On Debian/Ubuntu this\n"
                 f"usually means that the venv module is missing. Install it with:\n"
                 f"    sudo apt install python{version[0]}.{version[1]}-venv\n"
                 f"and run this script again.")

# 3. Install the packages with the environment's own Python
subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip"])
if latest:
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", *PACKAGES])
else:
    subprocess.check_call([python, "-m", "pip", "install", "-r", repo / "requirements.txt"])

# 4. Check that the code runs
subprocess.check_call([python, repo / "main.py", "--help"], stdout=subprocess.DEVNULL)

activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
print("\nDone! The environment works. From the repository folder, activate it with:")
print(f"    {activate}")
print("and run, for example:")
print("    python main.py --no_HJ --which_planets my_own")
