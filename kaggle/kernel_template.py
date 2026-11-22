"""Kaggle kernel entry point. Generated copies are pushed by scripts/kaggle_launch.py.

Unpacks the code bundle, installs dependencies, fetches public data from Hugging
Face, runs the listed commands, and leaves runs/ and the VLM response cache in
/kaggle/working so they can be pulled back with `kaggle kernels output`.
"""

import glob
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time

JOB = __JOB__  # noqa: F821  (filled in by scripts/kaggle_launch.py)

WORK = "/kaggle/working"
REPO = "/kaggle/tmp_repo"
T0 = time.time()


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    code = subprocess.call(cmd, shell=True)
    print(f"[exit {code}, t+{time.time() - T0:.0f}s]", flush=True)
    if check and code != 0:
        raise SystemExit(f"command failed: {cmd}")
    return code


# Kaggle may deliver the bundle as the tarball or already unpacked.
tarballs = glob.glob("/kaggle/input/**/reimagine_src.tar.gz", recursive=True)
markers = glob.glob("/kaggle/input/**/BUNDLE_COMMIT", recursive=True)
if tarballs:
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(tarballs[0]) as tar:
        tar.extractall(REPO)
elif markers:
    shutil.copytree(os.path.dirname(markers[0]), REPO)
else:
    subprocess.call("find /kaggle/input -maxdepth 4 | head -100", shell=True)
    raise SystemExit("code bundle not found under /kaggle/input")
os.chdir(REPO)
os.environ["MINDSEYE_GIT_COMMIT"] = open("BUNDLE_COMMIT").read().strip()
os.environ["HF_HOME"] = "/kaggle/hf_home"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("commit", os.environ["MINDSEYE_GIT_COMMIT"], flush=True)
if JOB.get("expect_commit") not in (None, os.environ["MINDSEYE_GIT_COMMIT"]):
    raise SystemExit(f"stale code bundle: expected {JOB['expect_commit']}")

# Fail in seconds, not after 20 minutes of retries, if Kaggle dropped the GPU or
# internet setting (it does so silently for accounts that are not phone-verified).
if sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False) != 0:
    raise SystemExit("no GPU in this session")
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as err:
    raise SystemExit(f"no internet in this session: {err}") from err
sh("df -h /kaggle /kaggle/working /tmp | cat", check=False)
sh(f"{sys.executable} -m pip install -q {JOB['pip']}")
sh(f"{sys.executable} -m pip install -q -e . --no-deps")
sh(
    f'{sys.executable} -c "import torch, transformers; '
    "print('torch', torch.__version__, 'transformers', transformers.__version__, "
    "'gpus', torch.cuda.device_count())\""
)

# Carry over runs and cached responses from an earlier session, if attached.
for prev in glob.glob("/kaggle/input/**/carryover", recursive=True):
    for name in ("runs", "cache"):
        if os.path.isdir(f"{prev}/{name}"):
            shutil.copytree(f"{prev}/{name}", f"{WORK}/{name}", dirs_exist_ok=True)
os.makedirs(f"{WORK}/runs", exist_ok=True)
os.makedirs(f"{WORK}/cache/vlm", exist_ok=True)

for cmd in JOB["setup"]:
    sh(cmd)
# Kaggle does not enforce the push timeout, and a cancelled session saves no output, so
# the kernel stops itself: each command is killed when the job's own deadline arrives,
# leaving 10 minutes for Kaggle to save /kaggle/working.
DEADLINE = T0 + JOB["timeout_s"] - 600
for cmd in JOB["commands"]:
    left = int(DEADLINE - time.time())
    if left <= 60:
        print(f"deadline reached; skipped: {cmd[:120]}", flush=True)
        continue
    # A failed job should not lose the outputs of the ones before it.
    full = shlex.quote(cmd.format(py=sys.executable, work=WORK))
    code = sh(f"timeout -s INT -k 60 {left} bash -c {full}", check=False)
    if code == 124:
        print("command stopped at the job deadline", flush=True)
print(f"total wall time {time.time() - T0:.0f}s", flush=True)
