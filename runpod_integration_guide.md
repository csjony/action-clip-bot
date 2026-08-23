# RunPod & Bot Integration Guide

This guide documents how the **Action Clip Bot** connects to the **RunPod GPU VPS** to generate videos using the **Wan 2.2 model** using a **Docker-free, zero-maintenance setup**.

---

## 1. Connection Topology & Port Architecture

```
+------------------+                   +----------------------------------+
|                  |                   | RunPod GPU Instance              |
|  Local Bot       |                   | (Docker Container)               |
|  (Orchestration) |                   |                                  |
|                  |                   |  +-----------------------------+ |
|                  |                   |  | JupyterLab Server           | |
|                  |                   |  | (Port 8888)                 | |
|                  |                   |  +-----------------------------+ |
|                  |                   |                                  |
|  Polling /ready  |  HTTP (Port 8000) |  +-----------------------------+ |
|  Submit Job      +------------------>|  | FastAPI gpu_server.py       | |
|  Fetch Video     |  Proxy/Tunnel URL |  | (Port 8000)                 | |
|                  |                   |  +-----------------------------+ |
|                  |                   |                                  |
|                  |                   |  +-----------------------------+ |
|                  |                   |  | Persistent venv             | |
|                  |                   |  | (/workspace/venv)           | |
|                  |                   |  +-----------------------------+ |
+------------------+                   +----------------------------------+
```

### Port Mapping Details:
*   **Port 8888 (JupyterLab):** Used for manual access, file uploads, and interactive terminal.
*   **Port 8000 (FastAPI Server):** Exposes the API endpoint for video generation. 
    *   `/ready` (GET): Tells the bot if the model is fully loaded in VRAM and ready for inference.
    *   `/generate` (POST): Queues an asynchronous generation job (returns a `job_id`).
    *   `/status/{job_id}` (GET): Checks whether the generation is queued, running, done, or failed.
    *   `/result/{job_id}` (GET): Downloads the completed video file.

---

## 2. Dynamic Persistent Virtual Environment Setup

To avoid requiring Docker builds on your local laptop, we use the official RunPod PyTorch base image:
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04`

### How It Works:
1. **Persistent Volume:** RunPod mounts your network volume at `/workspace`. Everything in this directory persists even when pods are terminated or recreated.
2. **First Boot (5-6 min):** The GPU server script detects that `/workspace/venv` does not exist. It automatically creates a python virtual environment there, installs all required libraries (like `diffusers`, `transformers`, etc.), and re-executes itself inside the venv.
3. **Subsequent Boots (under 10 seconds):** The GPU server script detects that `/workspace/venv` already exists, skips installation entirely, and executes instantly.
4. **Auto-Start Loop:** The pod is configured with a container start command that waits for `gpu_server.py` to be uploaded to `/workspace`. Once the file is present, it launches uvicorn automatically.

---

## 3. One-Time Setup Steps (How to Start)

Follow these steps once to initialize your RunPod GPU resources:

### Step A: Create a Network Volume on RunPod
1. Go to the **RunPod Console → Storage → Network Volumes**.
2. Click **Create Volume**.
3. Choose the same datacenter region you want to use for your GPUs.
4. Allocate a minimum of **50 GB** (needed to store the Hugging Face model cache and virtual environment).
5. Name it (e.g., `action-clip-bot-vol`) and note the volume ID (e.g. `vol-xyz123` or similar).

### Step B: Create your GPU Pod
1. Go to **RunPod → Pods** and click **Deploy**.
2. Select a GPU (Recommended: **NVIDIA H200** or **NVIDIA GeForce RTX 4090**).
3. Under **Template**, select the official **RunPod PyTorch** image (e.g., `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04`).
4. Click **Customize Deployment**:
   * **Container Disk:** `30 GB`
   * **Volume Mount Path:** `/workspace`
   * **Expose Ports:** Add `8000` (HTTP) and `8888` (HTTP) and `22` (TCP).
   * **Container Start Command:** Paste this exactly:
     ```bash
     bash -c 'while [ ! -f /workspace/gpu_server.py ]; do sleep 5; done; python /workspace/gpu_server.py'
     ```
   * **Environment Variables:** Add:
     - `WAN_MODEL_ID` = `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
     - `HF_HOME` = `/workspace/huggingface`
     - `DEPS_PREINSTALLED` = `0`
5. Connect your created **Network Volume** to this pod.
6. Deploy and start the pod.

### Step C: Upload the Server Script
1. Once the pod status is **RUNNING**, click **Connect** on the pod in the RunPod Console.
2. Click **Connect to JupyterLab** (opens in your browser on port 8888).
3. Drag and drop the file `gpu_server.py` from your local computer into the Jupyter file browser (`/workspace` folder).
4. The background start command will immediately detect the file and trigger uvicorn + the virtual environment setup.

You can view the setup logs inside JupyterLab by opening a terminal and checking the server output, or by checking the pod logs in the RunPod console.

---

## 4. Local Bot Configuration

Once the pod is initialized and you have noted the ID, configure your local bot `.env` file:

```ini
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_POD_ID=your_created_pod_id
RUNPOD_NETWORK_VOLUME_ID=your_network_volume_id
RUNPOD_GPU_TYPE="NVIDIA H200"  # or NVIDIA GeForce RTX 4090
RUNPOD_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04
```

Now, when you run `scripts/run_job.sh --quick-run`, the bot will:
1. Turn on the pod via the API.
2. Wait for the server to load the model (takes ~5-15 mins on first run to download model weights, then ~1 min on subsequent runs).
3. Generate the video clips.
4. Turn off the pod automatically to stop billing.
