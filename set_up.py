"""
Create a Python environment with everything the interpolator needs.

HOW TO USE

1. Create the environment (only once). From the repository folder:
       python3 set_up.py
   This creates the folder .venv in the repository, installs the package
   versions used in the paper (PACKAGES below) and checks that the code
   runs. With these versions, the default random seed reproduces the results
   of the paper exactly. The script can also be run from any other folder,
   e.g. python3 path/to/set_up.py.

2. Activate the environment (every time you open a new terminal):
       source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
   and run the code, e.g.
       python main.py --help
       python main.py --no_HJ --which_planets my_own
   Type "deactivate" to leave the environment. Without activating it, you
   can also run .venv/bin/python main.py (Windows: .venv\\Scripts\\python main.py).

OPTIONS AND COMMON PROBLEMS

- The package versions of the paper need Python 3.11 or 3.12. If your
  default Python is another version but one of these is installed, run the
  script with it, e.g. python3.12 set_up.py.
- python3 set_up.py --latest installs the newest package versions instead
  (any Python >= 3.10). The last digits of the results may then differ
  slightly from the paper.
- On Debian/Ubuntu, if creating the environment fails, install the venv
  module first (e.g. sudo apt install python3.12-venv) and run the script
  again.
- Running the script again reuses the existing environment. To start from
  scratch, delete the folder .venv first.

LaTeX is optional: if it is installed, it is used for the plot labels.
"""
import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# Package versions used for the results of the paper. With these versions,
# the default random seed reproduces the results exactly.
PACKAGES = {
    "numpy": "1.26.4",
    "scipy": "1.16.2",
    "matplotlib": "3.10.6",
    "tqdm": "4.67.1",
    "emcee": "3.1.6",
    "odrpack": "0.5.0",
}

repo = Path(__file__).resolve().parent
env = repo / ".venv"
python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        venv.create(env, with_pip=True, prompt="hj_magnetic_env")
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
    subprocess.check_call([python, "-m", "pip", "install",
                           *[f"{name}=={number}" for name, number in PACKAGES.items()]])

# 4. Check that the code runs
subprocess.check_call([python, repo / "main.py", "--help"], stdout=subprocess.DEVNULL)

activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
print("\nDone! The environment works. From the repository folder, activate it with:")
print(f"    {activate}")
print("and run, for example:")
print("    python main.py --no_HJ --which_planets my_own")
