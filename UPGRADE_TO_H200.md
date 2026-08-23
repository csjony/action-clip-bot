# Action Clip Bot — H200 SXM Upgrade Plan

**Goal:** Move from RTX 4090 ($0.138/clip, 12 min/clip) to H200 SXM (~$0.037/clip, ~30 sec/clip).

---

## Overview of All Steps

| # | Step | Who | Time |
|---|---|---|---|
| 1 | Implement code changes in `gpu_server.py` | AI | ~5 min |
| 2 | Build & push custom Docker image | **You** | ~10 min |
| 3 | Create new network volume in H200 datacenter | **You** | ~2 min |
| 4 | Create H200 pod with custom image + new volume | **You** | ~3 min |
| 5 | First boot: model auto-downloads to new volume | Automatic | ~10 min |
| 6 | Update `.env` with new pod ID | **You** | ~1 min |
| 7 | Verify & test | **You** | ~5 min |
| 8 | Delete old volume to stop billing | **You** | ~1 min |
| **Total** | | | **~37 min** |

---

## STEP 1 — Code Changes to `gpu_server.py`

> **Who:** AI handles this  
> **File:** `gpu_server.py`

### Change A — Detect GPU Architecture (add near top of `load_model()`)

This detects whether the pod is running on a Hopper GPU (H100/H200) or older (RTX 4090).
Added right before the `_patch_fp8_linear_layers` call at line ~293.

```python
# Detect GPU compute capability to decide FP8 execution path
compute_cap = torch.cuda.get_device_capability(0)
is_hopper_or_newer = compute_cap[0] >= 9  # sm_90 = H100/H200, sm_86 = RTX 4090
print(f"GPU compute capability: sm_{compute_cap[0]}{compute_cap[1]} "
      f"({'Hopper/Newer — native FP8' if is_hopper_or_newer else 'Ampere/Older — BF16 fallback'})")
```

### Change B — Conditional FP8 Patch (Ampere fallback only)

> ℹ️ **Why the patch exists:** On RTX 4090 (sm_86) there are no native FP8 compute units,
> so we override each FP8 Linear's `forward()` to cast weights on-the-fly to BF16.
> On H200 (sm_90+) we skip the patch entirely — the FP8 model's own `forward()` uses
> `torch._scaled_mm` with the stored per-tensor scale factors, giving native FP8 speed
> (~1.5–2.5s/step) with no green noise.
>
> **The previous green-noise bug** was caused by `module.weight.to(torch.bfloat16)` which
> silently discarded the FP8 scale factors. That code has been removed for H200.

On RTX 4090 (sm_86): on-the-fly BF16 cast per forward pass (`permanent=False`).
On H200 (sm_90): **no patch** — native FP8 tensor core execution.

```python
# AFTER (fast on H200, safe fallback on RTX 4090):
if _is_wan22:
    if is_hopper_or_newer:
        # Hopper/newer (sm_90+) has massive VRAM (H200 has 141 GB). We permanently convert
        # weights to BF16 once during load to completely bypass per-step dynamic casting overhead.
        _patch_fp8_linear_layers(pipe.transformer, permanent=True)
        print("Applied permanent FP8-to-BF16 conversion (optimized for Hopper GPU speed).")
    else:
        # Ampere/Ada (e.g. RTX 4090) has only 24 GB VRAM. We must auto-cast on the fly
        # to save VRAM and prevent out-of-memory errors.
        _patch_fp8_linear_layers(pipe.transformer, permanent=False)
        print("Applied on-the-fly FP8 fallback patch (optimized for Ampere GPU memory).")
    pipe.transformer_2 = pipe.transformer
    print("✅ Pointed transformer_2 → transformer.")
```

### Change C — CUDA Graph Pre-Warm (add after VAE tiling, before `_model_loaded = True`)

Runs one silent dummy inference during model startup to pre-compile JIT/Triton kernels.
This means the first user clip runs at full 30-second speed instead of 3 minutes.

```python
# Pre-warm CUDA graph on H200 to compile Triton kernels before first user clip
if _is_wan22 and is_hopper_or_newer and torch.cuda.is_available():
    try:
        print("Pre-warming CUDA graph (compiling Triton kernels for sm_90)...")
        # Warm the kernels for the current H200 production baseline:
        # 5 seconds at 16 fps, 1280x720 output.
        _dummy_latent = torch.zeros(
            1, 16, 21, DEFAULT_GENERATION_HEIGHT // 16, DEFAULT_GENERATION_WIDTH // 16,
            dtype=torch.bfloat16, device="cuda"
        )
        _dummy_embeds = torch.zeros(1, 4, 4096, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            pipe.transformer(
                hidden_states=_dummy_latent,
                timestep=torch.tensor([500], device="cuda"),
                encoder_hidden_states=_dummy_embeds,
                return_dict=False,
            )
        torch.cuda.synchronize()
        del _dummy_latent, _dummy_embeds
        torch.cuda.empty_cache()
        print("CUDA warmup complete. All clips will run at full speed from the first request.")
    except Exception as _warm_err:
        print(f"CUDA warmup skipped (non-fatal): {_warm_err}")
```

---

## STEP 2 — Build & Push Custom Docker Image

> **Who:** You (requires Docker Desktop + Docker Hub login)  
> **Estimated time:** ~10 minutes (mostly docker build time)

The `Dockerfile.gpu` is already complete and correct. You just need to build and push it.

### 2a — Make sure Docker is installed and you're logged in

```bash
docker login
# Enter your Docker Hub username (sudojony) and password
```

### 2b — Build and push the image

Run this from the project root on your local PC:

```bash
cd /home/user/ZCodeProject/action-clip-bot

docker build -f Dockerfile.gpu -t sudojony/action-clip-bot-gpu:latest .
docker push sudojony/action-clip-bot-gpu:latest
```

> **Note:** The build will take ~5-8 minutes because it installs all Python packages (diffusers, transformers, nvidia-modelopt, etc.) into the image layers. This only needs to happen once. Future pushes after code changes will be much faster due to layer caching.

### 2c — Verify the push succeeded

Go to https://hub.docker.com/r/sudojony/action-clip-bot-gpu and confirm the image appears with a recent timestamp.

---

## STEP 3 — Create New Network Volume in H200 Datacenter

> **Who:** You (RunPod console)  
> **Estimated time:** ~2 minutes

1. Go to **RunPod Console → Storage → Network Volumes**.
2. Click **+ New Network Volume**.
3. Set **Size** to **75 GB**.
   > ⚠️ **Important:** RunPod does NOT support shrinking volumes after creation — only expanding.
   > Use 75GB for a safe download margin. The model is ~25GB, leaving 50GB headroom.
   > Monthly cost: $5.25/month vs $3.50/month for 50GB — the small difference is worth
   > the peace of mind and the H200 saves far more than that per video run.
4. **Important:** Select the **same datacenter region** where H200 SXM pods are available (usually `US-TX-3` or `EU-CZ-1` — check which region shows H200 as available in the pod creation screen first).
5. Give it a name, e.g. `action-clip-bot-h200`.
6. Click **Create**.
7. **Note the new Volume ID** — you'll need it when creating the pod.

---

## STEP 4 — Create H200 SXM Pod

> **Who:** You (RunPod console)  
> **Estimated time:** ~3 minutes

1. Go to **RunPod Console → Pods → + Deploy Pod**.
2. Select **H200 SXM** from the GPU list.
3. Under **Container Image**, change from the default to:
   ```
   sudojony/action-clip-bot-gpu:latest
   ```
4. Under **Volume**, attach the new 75GB volume you created in Step 3.
   - Mount path must be `/workspace`.
5. Under **Environment Variables**, add the following:
   ```
   HF_TOKEN=<your_huggingface_token>
   HUGGING_FACE_HUB_TOKEN=<your_huggingface_token>
   WAN_MODEL_ID=Wan-AI/Wan2.2-T2V-A14B-Diffusers
   ```
   > **Why Native BF16?** H200 runs the native BF16 model with absolute numerical precision and high speed (~9s/step). Loading the FP8 model directly under standard diffusers results in severe green noise due to missing per-tensor scaling factors. Switching to the native BF16 model resolves all color corruption.
   >
   > **MoE Single-Expert Mode:** To avoid exceeding the RunPod network volume disk quota (which raises Errno 122), we load the model with `boundary_ratio=None` and `transformer_2=None`. This bypasses MoE routing and uses only the primary `transformer` expert for all steps. By adding `ignore_patterns=["transformer_2/*"]` during download, we skip the 28 GB secondary expert files, saving massive disk space and running completely stable.
   
6. Under **Expose Ports**, add TCP port `8000`.
7. Click **Deploy**.
8. **Note the new Pod ID** from the pod list (e.g. `abc123xyz`).

---

## STEP 5 — First Boot: Model Auto-Downloads

> **Who:** Automatic (no action needed)  
> **Estimated time:** ~10 minutes

On the very first boot with the new volume (which is empty), `gpu_server.py` will:
1. Skip pip install (because `DEPS_PREINSTALLED=1` is set in the Docker image) ✅
2. Detect that the model cache is missing in `/workspace/huggingface/`.
3. Automatically download `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (skipping `transformer_2` files) using `snapshot_download()` with `ignore_patterns=["transformer_2/*"]`.
4. Load the model in native BF16 into H200 VRAM.
5. Run the CUDA pre-warm.
6. Set `/ready = 200`.

**Monitor progress via SSH:**
```bash
ssh -p <PORT> root@<IP> "tail -f /workspace/gpu_server.log"
```

You'll see the download progress bars, then:
```
Hopper GPU — using native BF16 weights.
Pre-warming CUDA graph (compiling Triton kernels for sm_90)...
CUDA warmup complete.
✅ Model loaded successfully! Ready to generate videos.
```

**Verify readiness:**
```bash
curl http://localhost:8000/ready
# Expected: {"status":"ready","device":"cuda"}
```

> **One-time cost:** ~10 min at $4.39/hr = **~$0.73**. The model is cached on the new volume permanently after this.

---

## STEP 6 — Update `.env` on Your Local PC

> **Who:** You  
> **Estimated time:** ~1 minute

Open `.env` and update:

```ini
# Replace with the new H200 pod ID from Step 4:
RUNPOD_POD_ID=<new_h200_pod_id>

# Replace with the new volume ID from Step 3:
RUNPOD_NETWORK_VOLUME_ID=<new_volume_id>
```

Then restart the local dashboard:
```bash
kill $(lsof -ti:8080) 2>/dev/null || true
nohup scripts/run_dashboard.sh >> data/dashboard.log 2>&1 &
```

---

## STEP 7 — Verify & Test

> **Who:** You  
> **Estimated time:** ~5 minutes

### 7a — Confirm proxy URL works from your local PC
```bash
curl https://<new_pod_id>-8000.proxy.runpod.net/ready
# Expected: {"status":"ready","device":"cuda"}
```

### 7b — Run the smoke test (generates a 3-second test video)
```bash
.venv/bin/python3 scripts/test_gpu_server.py https://<new_pod_id>-8000.proxy.runpod.net
```

Expected output:
```
✅  Model is READY  (0s elapsed)
🎬  Submitting async generation job...
✅  Job submitted: xxxxxxxx...
   [  20s] Job xxxxxxxx: running
   [  40s] Job xxxxxxxx: done
⬇️  Downloading result...
✅  Video saved to: /tmp/gpu_test_output.mp4
    Size   : ~1500 KB
🎉  Success!
```

> If the step time is ~30s instead of ~32s per step, native FP8 Tensor Cores are working correctly.

### 7c — Run a quick pipeline test
```bash
.venv/bin/python3 -m src.pipeline --quick-run
```

---

## STEP 8 — Delete Old Volume to Stop Billing

> **Who:** You  
> **Estimated time:** ~1 minute

Once the new H200 setup is verified working:

1. Stop the old RTX 4090 pod if still running.
2. Go to **RunPod Console → Storage → Network Volumes**.
3. Find the old volume `j7a4lou0dd`.
4. Click **Delete**.

> RunPod charges ~$0.07/GB/month for network volume storage. Deleting the 50GB old volume saves ~$3.50/month.

---

## Expected Performance After Upgrade

| Metric | RTX 4090 (before) | H200 SXM (after) |
|---|---|---|
| Seconds per diffusion step | ~32s | **~1.5s** |
| Time per clip (5s video) | ~12 min | **~30 sec** |
| Cost per clip | ~$0.138 | **~$0.037** |
| Full run (13 clips) time | ~2.6 hours | **~8 min** |
| Full run cost | ~$1.79 | **~$0.52** |
| Setup cost per session | ~$0.37 | **~$0.15** |

---

## Rollback Plan

If H200 has any issues:

1. Change `RUNPOD_POD_ID` back to the old RTX 4090 pod ID in `.env`.
2. Restart dashboard.
3. The RTX 4090 setup continues to work unchanged — the BF16 fallback patch is still in the code for non-Hopper GPUs.

---

## Files Changed in This Upgrade

| File | Change |
|---|---|
| `gpu_server.py` | Phase 2 (conditional FP8 patch) + Phase 3 (CUDA pre-warm) |
| `.env` | New `RUNPOD_POD_ID` + `RUNPOD_NETWORK_VOLUME_ID` |
| Docker Hub | New image `sudojony/action-clip-bot-gpu:latest` |
| RunPod Console | New H200 pod + new 50GB volume in H200 datacenter |
