"""Run on the remote pod to inspect FP8 weight and scale factor structure."""
import os, sys
from src.generators.runpod_manager import RunPodManager
from dotenv import load_dotenv
load_dotenv()

manager = RunPodManager(os.getenv("RUNPOD_API_KEY"), os.getenv("RUNPOD_POD_ID"))
info = manager.get_pod_info()
ip = info.get("publicIp")
port = (info.get("portMappings") or {}).get("22")
if not ip or not port:
    print("Pod is stopped — start it first")
    sys.exit(1)

import subprocess
cmd = """python3 -c "
import torch
from safetensors import safe_open
import glob, os

# Find transformer safetensors
hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
pattern = hf_home + '/hub/models--nvidia--Wan2.2-T2V-A14B-Diffusers-FP8/snapshots/*/transformer/*.safetensors'
files = sorted(glob.glob(pattern))
print(f'Found {len(files)} transformer shard(s)')

if files:
    with safe_open(files[0], framework='pt', device='cpu') as f:
        keys = list(f.keys())
        fp8_keys = [k for k in keys if 'weight' in k][:5]
        scale_keys = [k for k in keys if 'scale' in k.lower() or 'amax' in k.lower()][:10]
        print('\\nSample weight keys:', fp8_keys)
        print('Scale/amax keys (first 10):', scale_keys)
        if fp8_keys:
            w = f.get_tensor(fp8_keys[0])
            print(f'\\nWeight dtype: {w.dtype}, shape: {w.shape}, min: {w.float().min():.4f}, max: {w.float().max():.4f}')
        if scale_keys:
            s = f.get_tensor(scale_keys[0])
            print(f'Scale dtype: {s.dtype}, value: {s}')
"
"""
result = subprocess.run(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
     "-p", str(port), f"root@{ip}", cmd],
    capture_output=True, text=True, timeout=30
)
print(result.stdout or result.stderr)
