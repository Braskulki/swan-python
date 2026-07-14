import subprocess
import os

base = os.path.dirname(os.path.dirname(__file__))
swan_dir = os.path.join(base, "swan")
input_file = os.path.join(swan_dir, "input")


subprocess.run(
    ["swanrun", input_file],
    shell=True,
)