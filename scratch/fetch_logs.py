import os
import sys
from pathlib import Path
from src.generators.runpod_manager import RunPodManager

# Load env variables
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("RUNPOD_API_KEY")
pod_id = os.getenv("RUNPOD_POD_ID")

if not api_key or not pod_id:
    print("Missing RUNPOD_API_KEY or RUNPOD_POD_ID in .env")
    sys.exit(1)

manager = RunPodManager(api_key, pod_id)
pod_info = manager.get_pod_info()
public_ip = pod_info.get("publicIp")
port_mappings = pod_info.get("portMappings") or {}
ssh_port = port_mappings.get("22")

if not public_ip or not ssh_port:
    print("Could not find public IP or SSH port. Pod might be stopped.")
    # Try starting the pod or checking status
    print("Status is:", manager.get_status())
    sys.exit(1)

print(f"Connecting to {public_ip}:{ssh_port} via SSH...")
import subprocess
ssh_cmd = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-p", str(ssh_port),
    f"root@{public_ip}",
    "tail -n 150 /workspace/gpu_server.log"
]
res = subprocess.run(ssh_cmd, capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)
