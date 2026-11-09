"""Run SenseNova-U1 image self-correction with profiling enabled.

Pass an HF checkpoint and either --prompt or --jsonl. Thinking is disabled
by default. Text-only samples default to 2048×2048. Use --width and --height
or --resolution to change it.
For text-only JSONL benchmarks, use one {"prompt": "..."} object per line,
with optional width and height. Image paths in a sample are input images.
"""

import sys

from inference import main as run_interleave

from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

if __name__ == "__main__":
    user_args = sys.argv[1:]
    defaults = ["--no-think_mode", "--system_message", SYSTEM_MESSAGE_FOR_GEN, "--profile"]

    if not any(arg.partition("=")[0] == "--resolution" for arg in user_args):
        defaults.extend(["--width", "2048", "--height", "2048"])

    if "-h" in user_args or "--help" in user_args:
        print(__doc__)

    sys.argv[1:] = defaults + user_args
    run_interleave()
