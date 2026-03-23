from pathlib import Path
import subprocess
import argparse
import sys

import qtoc_krylov.utilities.paths as paths

# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=paths.CONFIG_DIR+'defaults.yml')
args = parser.parse_args()

scripts = [
        "reproduce_fig_1_and_2.py",
        "reproduce_fig_3.py",
        "reproduce_fig_4_and_5.py",
        "reproduce_fig_6.py",
        "reproduce_fig_7.py",
        "reproduce_fig_8.py"
        ]

dir = Path(__file__).parent

for script in scripts:
    script_path = str(dir / script)

    print(f"--- Running {script} ---")
    result = subprocess.run([sys.executable, script_path,
                             f'--config={args.config}'])

    if result.returncode != 0:
        print(f'Error while running {script}')
        sys.exit(result.returncode)

print("Finished.")
