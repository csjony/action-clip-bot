# =============================================================================
# Cloud GPU / VPS Wan 2.2 Video Generation API Server
# 
# How to run:
# 1. Run this script on a GPU VPS (e.g. RunPod, Lambda Labs, etc.)
# 2. Set environment variables (e.g. WAN_MODEL_ID)
# 3. Expose port 8000
# 4. Upload this file (gpu_server.py) and execute it
# 5. Copy the public URL printed at the end and put it in your bot's accounts or env
# =============================================================================

import sys
import os
import subprocess
import time

# Auto-route Hugging Face model caching to RunPod's persistent volume if available
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/huggingface")
    print("Detected RunPod persistent storage. Setting HF_HOME=/workspace/huggingface")

# Reduce CUDA memory fragmentation — critical for fitting FP8 model + inference on 32 GB VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

_debug_info = {
    "status": "not started",
    "server_version": "v40-h200-primary",  # bump this on each deploy to confirm GPU VPS is on latest code
    "text_encoder_status": "not loaded",
    "text_encoder_error": None,
    "pipe_load_status": "not loaded",
    "pipe_load_error": None,
    "last_generation_error": None,
    "versions": {}
}

def _system_deps_ok() -> bool:
    """True when the current interpreter can run the server as-is.

    Checks every module imported at server top level. Transformers must be
    4.x — 5.x broke the bitsandbytes layer-wise dequantization the FP8
    fallback relies on (hence the 4.57.6 pin elsewhere).
    """
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import nest_asyncio  # noqa: F401
        if not transformers.__version__.split(".")[0] == "4":
            print(f"System transformers {transformers.__version__} is not 4.x — using venv path.")
            return False
        return True
    except ImportError as exc:
        print(f"System python missing {exc} — using venv path.")
        return False


def setup_environment():
    # 0. If the SYSTEM interpreter already satisfies every runtime import,
    # use it directly — no venv, no pip, no re-exec. This is the normal
    # Colab case (the notebook cell installs into system python), and it
    # also sidesteps systems whose python cannot build venvs at all
    # (missing ensurepip → "No module named pip" death spiral).
    if _system_deps_ok():
        print("System python already has all dependencies — skipping venv (fast path).")
        import diffusers, transformers, accelerate, bitsandbytes
        print(f"System versions:\n"
              f"  diffusers: {diffusers.__version__}\n"
              f"  transformers: {transformers.__version__}\n"
              f"  accelerate: {accelerate.__version__}\n"
              f"  bitsandbytes: {bitsandbytes.__version__}")
        _debug_info["versions"] = {
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__
        }
        return

    # 1. If running from our pre-built Docker image, all deps are already installed —
    # skip the slow pip install.
    if os.environ.get("DEPS_PREINSTALLED") == "1":
        print("DEPS_PREINSTALLED=1 detected — skipping pip install (using baked image deps).")
        import diffusers, transformers, accelerate, bitsandbytes
        print(f"Pre-installed versions:\n"
              f"  diffusers: {diffusers.__version__}\n"
              f"  transformers: {transformers.__version__}\n"
              f"  accelerate: {accelerate.__version__}\n"
              f"  bitsandbytes: {bitsandbytes.__version__}")
        _debug_info["versions"] = {
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__
        }
        return

    # 2. Use a virtual environment on the LOCAL container disk (/root/venv) to avoid
    # the extreme slowness of creating thousands of small Python package files on
    # the network-mounted /workspace filesystem (MooseFS).
    # NOTE: Model weights still go to /workspace/huggingface — large sequential
    # reads/writes are fine on network storage, but pip's many tiny file ops are not.
    if not os.environ.get("IN_VENV"):
        venv_path = "/root/venv"
        venv_python = os.path.join(venv_path, "bin", "python")
        # Use a sentinel file to confirm pip install completed — not just the python binary,
        # which is created by `venv` before packages are installed (causing empty-venv re-exec).
        sentinel = os.path.join(venv_path, ".deps_installed")
        if not os.path.exists(venv_python) or not os.path.exists(sentinel):
            # A stale venv without a working pip (interrupted creation, or a
            # system python whose venv ships without pip — seen on Colab as
            # "/root/venv/bin/python: No module named pip") poisons every
            # step below. Detect and rebuild instead of crashing.
            if os.path.exists(venv_python):
                try:
                    subprocess.run([venv_python, "-m", "pip", "--version"],
                                   capture_output=True, check=True)
                except Exception:
                    print("Stale venv without working pip — removing and recreating...")
                    import shutil
                    shutil.rmtree(venv_path, ignore_errors=True)
            if not os.path.exists(venv_python):
                print("Creating virtual environment on local disk at /root/venv...")
                subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv_path], check=True)
                try:
                    subprocess.run([venv_python, "-m", "pip", "--version"],
                                   capture_output=True, check=True)
                except Exception:
                    print("Fresh venv has no pip — bootstrapping via ensurepip...")
                    subprocess.run([venv_python, "-m", "ensurepip", "--upgrade"], check=True)
            print("Upgrading pip...")
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
            print("Installing dependencies (this takes ~3 min on first boot)...")
            subprocess.run([
                venv_python, "-m", "pip", "install", "-q",
                "diffusers",
                "transformers==4.57.6",  # Pin to 4.x — transformers 5.x broke bitsandbytes layer-wise dequantization
                "accelerate", "fastapi", "uvicorn", "nest_asyncio", "httpx",
                "bitsandbytes>=0.43.0", "hf_transfer",
                "imageio", "imageio-ffmpeg",
                "nvidia-modelopt[hf]",
                "opencv-python-headless"
            ], check=True)
            # Write sentinel only after successful install
            open(sentinel, "w").close()
            print("Dependencies installed successfully.")

        print("Re-executing inside /root/venv...")
        os.environ["IN_VENV"] = "1"
        os.execv(venv_python, [venv_python] + sys.argv)
        sys.exit(0)

    # If we are already running inside our venv, skip system pip upgrades
    if os.environ.get("IN_VENV") == "1":
        return

    # 3. Fallback: Install/upgrade in system environment if /workspace not present
    print("Installing/upgrading python dependencies in system environment...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U",
         "diffusers",
         "transformers==4.57.6",
         "accelerate", "fastapi", "uvicorn", "nest_asyncio", "httpx",
         "bitsandbytes>=0.43.0", "hf_transfer",
         "nvidia-modelopt[hf]",
         "opencv-python-headless"],
        check=True
    )
    print("Dependencies upgraded successfully.")
    import diffusers
    import transformers
    import accelerate
    import bitsandbytes
    print(f"Installed versions:\n"
          f"  diffusers: {diffusers.__version__}\n"
          f"  transformers: {transformers.__version__}\n"
          f"  accelerate: {accelerate.__version__}\n"
          f"  bitsandbytes: {bitsandbytes.__version__}")
    _debug_info["versions"] = {
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__
    }

# Run setup in environment
setup_environment()

import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# ── Monkey-patch: fix diffusers bug loading nvidia/Wan2.2-T2V-A14B-Diffusers-FP8 ──
# The model's config.json sets "quant_method":"modelopt" but omits "quant_type".
# diffusers tries to instantiate NVIDIAModelOptConfig(**config_dict) which raises:
#   TypeError: NVIDIAModelOptConfig.__init__() missing 1 required positional argument: 'quant_type'
# We intercept the constructor and default quant_type to "FP8" when it is absent.
try:
    from diffusers.quantizers.quantization_config import NVIDIAModelOptConfig as _NMC
    
    # Patch 1: Missing quant_type default
    _orig_nmc_init = _NMC.__init__
    def _patched_nmc_init(self, *args, **kwargs):
        if not args and "quant_type" not in kwargs:
            kwargs["quant_type"] = "FP8"
        _orig_nmc_init(self, *args, **kwargs)
    _NMC.__init__ = _patched_nmc_init
    
    # Patch 2: Fix TypeError: 'list' object is not a mapping in modelopt config resolution
    _orig_get_config = _NMC.get_config_from_quant_type
    def _patched_get_config_from_quant_type(self):
        try:
            return _orig_get_config(self)
        except TypeError as err:
            if "list" in str(err) and getattr(self, "quant_type", None) == "FP8":
                print("⚠️ Detected ModelOpt config compatibility drift. Falling back to default FP8 configuration.")
                return {
                    "quant_cfg": {
                        "*weight": {"num_bits": 8, "axis": 0, "type": "fp8"},
                        "*input": {"num_bits": 8, "axis": -1, "type": "fp8"},
                    },
                    "algorithm": getattr(self, "algorithm", "max"),
                }
            raise err
    _NMC.get_config_from_quant_type = _patched_get_config_from_quant_type
    
    print("✅ Applied NVIDIAModelOptConfig patches (quant_type default + get_config fallback).")
except Exception as _patch_err:
    print(f"⚠️  NVIDIAModelOptConfig patch skipped: {_patch_err}")

# Force execution device to always be CUDA if available, avoiding CPU fallback when text_encoder parameters are moved to CPU
if torch.cuda.is_available():
    WanPipeline._execution_device = property(lambda self: torch.device("cuda"))

from contextlib import asynccontextmanager

# Allow uvicorn to run inside notebook event loops
nest_asyncio.apply()

# Set before any CUDA allocation — reduces memory fragmentation during inference.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
_model_loaded = False
_model_loading = False
# Live load progress surfaced via /ready (503 detail), /health and /progress
# so dashboards show "downloading weights 42%" instead of a bare error.
_load_progress: dict = {"phase": "starting", "pct": 0.0, "detail": ""}


def _set_phase(phase: str, pct: float = 0.0, detail: str = "") -> None:
    _load_progress["phase"] = phase
    _load_progress["pct"] = pct
    _load_progress["detail"] = detail


_is_wan22 = False
_is_hopper_or_newer = False
_te_device = "cpu"
_use_cpu_offload = False
total_vram_gib = 0.0
DEFAULT_GENERATION_FPS = 16
DEFAULT_GENERATION_WIDTH = 1280
DEFAULT_GENERATION_HEIGHT = 720

import threading
import uuid
_jobs: dict = {}
_generation_lock = threading.Lock()
_uuid = uuid
_threading = threading

# ── Self-hosted Foley (MMAudio video-to-audio, free) ─────────────────────
# Only ONE heavy model family lives on GPU at a time. /generate keeps the Wan
# video pipe; /foley frees it and loads MMAudio small (≈3 GB weights,
# T4-friendly), and the next /generate transparently switches back. Each
# switch costs minutes of (re)load — so the pipeline renders ALL clips first
# and runs Foley once in the audio phase. NOTE: MMAudio is CC-BY-NC-4.0
# (non-commercial); monetized channels should prefer licensed SFX.
_foley = None                # dict(net, feature_utils, model) when loaded
_active_backend = "video"    # "video" | "foley"
_foley_lock = threading.Lock()


def _free_video_models() -> None:
    """Drop the Wan pipe so Foley fits in VRAM. Next /generate reloads it."""
    global pipe, _model_loaded, _model_loading, _active_backend
    pipe = None
    _model_loaded = False
    _model_loading = False
    _active_backend = "foley"
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Video models freed — Foley backend active.")


def _free_foley_models() -> None:
    global _foley, _active_backend
    _foley = None
    _active_backend = "video"
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Foley models freed — video backend active.")


def _ensure_video() -> None:
    """Switch back to Wan if Foley is resident; otherwise a no-op.

    Called at the top of every video job. When the video pipe is already
    loaded this costs nothing, so the normal (never-foley) path is unchanged.
    """
    global _active_backend
    if _active_backend == "video" and pipe is not None and _model_loaded:
        return
    if _active_backend == "foley":
        print("Switching video backend back on — freeing Foley, (re)loading Wan...")
        _free_foley_models()
    # load_model() is synchronous and skips itself when already loaded.
    load_model()


def _ensure_foley():
    """Load MMAudio (frees the Wan pipe first). Raises RuntimeError if the
    mmaudio package isn't installed on this server."""
    global _foley, _active_backend
    if _active_backend == "foley" and _foley is not None:
        return _foley
    t0 = time.time()
    print("Switching to Foley backend — freeing video models first...")
    _free_video_models()
    # Lazy imports: RunPod images without the mmaudio package must still
    # import this module cleanly; only /foley needs it.
    try:
        from mmaudio.eval_utils import ModelConfig, all_model_cfg
        from mmaudio.model.networks import MMAudio, get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils
    except ImportError as exc:
        raise RuntimeError(
            "MMAudio package not installed on this GPU server. "
            "Colab: add git+https://github.com/hkchengrex/MMAudio.git to pip install.") from exc
    variant = os.environ.get("MMAUDIO_VARIANT", "small_16k")
    if variant not in all_model_cfg:
        raise RuntimeError(
            f"Unknown MMAudio variant {variant!r}. Choices: {sorted(all_model_cfg)}")
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    dev = device
    dtype = torch.bfloat16
    net: MMAudio = get_my_mmaudio(model.model_name).to(dev, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=dev, weights_only=True))
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(dev, dtype).eval()
    _foley = {"net": net, "feature_utils": feature_utils, "model": model}
    print(f"Foley backend ready ({variant}, took {time.time() - t0:.0f}s).")
    return _foley

app = FastAPI(title="Wan GPU Server")


@app.on_event("startup")
async def startup_event():
    print("FastAPI server started. Starting model loading in a background thread...")
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=_clean_stale_jobs_loop, daemon=True).start()


def _clean_stale_jobs_loop():
    import time
    while True:
        try:
            time.sleep(60)
            now = time.time()
            stale_keys = []
            # Create a static snapshot list of items to avoid modification during iteration
            for job_id, job in list(_jobs.items()):
                # Clean up jobs older than 30 minutes (1800 seconds)
                if now - job.get("created_at", 0) > 1800:
                    stale_keys.append(job_id)
            
            for job_id in stale_keys:
                job = _jobs.pop(job_id, None)
                if job:
                    video_path = job.get("video_path")
                    if video_path and os.path.exists(video_path):
                        try:
                            os.unlink(video_path)
                            print(f"[cleanup] Deleted stale video file: {video_path}")
                        except Exception as e:
                            print(f"[cleanup] Failed to delete file {video_path}: {e}")
                    print(f"[cleanup] Removed stale job entry {job_id}")
        except Exception as e:
            print(f"[cleanup] Error in stale jobs cleanup loop: {e}")


def load_model():
    """Load Wan 2.2 model. Strategy is selected automatically based on GPU VRAM.

    PRIMARY PATH — H200 SXM 141 GB (>= 120 GB VRAM), native BF16 + FP8 tensor cores:
      Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers (no quantization)
      Both transformer experts (~56 GB) + text encoder (~22 GB BF16) + VAE (~2 GB)
      all loaded directly to GPU VRAM. CUDA warmup pre-compiles Triton kernels.
      Estimated ~8-12 s/denoising step → ~3-4 min/clip → ~40-50 min/full video.

    LARGE-VRAM PATH — A100 SXM 80 GB (60-120 GB VRAM), native BF16:
      Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers (no quantization)
      Both transformer experts + VAE on GPU. Text encoder (~22 GB) on CPU to free
      headroom for activations. CUDA warmup runs to pre-compile kernels.
      Estimated ~70 s/denoising step → ~23 min/clip.

    FALLBACK PATH — A40/A6000 (48 GB VRAM), native BF16:
      Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers (no quantization)
      accelerate enable_model_cpu_offload() manages expert swapping automatically.
      Peak VRAM stays within the 48 GB limit at the cost of PCIe transfer overhead.

    LOW-VRAM PATH — RTX 4090 (24 GB VRAM), FP8 model:
      Model: nvidia/Wan2.2-T2V-A14B-Diffusers-FP8
      The FP8 patch overrides each FP8 Linear.forward() to cast weights on-the-fly
      to the input dtype (BF16) with correct per-layer dequantization scale factors
      pre-loaded from the checkpoint safetensors.
    """
    global pipe, _model_loaded, _model_loading, _debug_info, _is_wan22, _is_hopper_or_newer, _te_device, _use_cpu_offload, total_vram_gib
    if _model_loaded or _model_loading:
        return
    _model_loading = True
    _debug_info["status"] = "loading"
    try:
        # Enable TensorFloat-32 (TF32) for faster float32 matrix multiplications on Ampere/Hopper
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            print("✅ TensorFloat-32 (TF32) matmul precision set to HIGH.")

        model_id = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        _is_wan22 = "wan2.2" in model_id.lower()
        print(f"Loading {model_id} (is_wan22={_is_wan22})...")

        # ── Model profile: VRAM floor depends on parameter count ──────────
        # 14B-class needs a datacenter GPU; 5B fits 24 GB cards; the 1.3B
        # model is the Colab-free-tier (T4 15 GB) option — lower fidelity,
        # but it runs. Override the floor with WAN_MIN_VRAM_GIB.
        _lower = model_id.lower()
        if "14b" in _lower or "t2v-a14b" in _lower:
            _min_vram = float(os.environ.get("WAN_MIN_VRAM_GIB", "20"))
            _profile = "14B"
        elif "5b" in _lower or "ti2v-5b" in _lower:
            _min_vram = float(os.environ.get("WAN_MIN_VRAM_GIB", "14"))
            _profile = "5B"
        else:
            _min_vram = float(os.environ.get("WAN_MIN_VRAM_GIB", "8"))
            _profile = "small (<=1.3B)"

        # ── VRAM preflight (fail fast with a clear message, don't OOM mid-load)
        # Refusing BEFORE the multi-GB download beats an OOM-kill with no
        # actionable error. Override with WAN_ALLOW_SMALL_GPU=1 to attempt
        # anyway, or point WAN_MODEL_ID at a smaller model.
        if torch.cuda.is_available():
            _vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            _allow_small = os.environ.get("WAN_ALLOW_SMALL_GPU", "").lower() in ("1", "true", "yes")
            print(f"Preflight: GPU VRAM {_vram_gib:.1f} GiB, profile {_profile}, needs >={_min_vram:.0f} GiB.")
            if _vram_gib < _min_vram and not _allow_small:
                raise RuntimeError(
                    f"GPU has only {_vram_gib:.1f} GiB VRAM but {model_id} "
                    f"({_profile} profile) needs >={_min_vram:.0f} GiB — loading "
                    f"would OOM. Use a bigger GPU, a smaller WAN_MODEL_ID "
                    f"(e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers for free Colab), "
                    f"or set WAN_ALLOW_SMALL_GPU=1 to attempt anyway (likely to crash).")
            print("✅ VRAM preflight passed.")
        else:
            print("⚠️  No CUDA GPU detected — model load will fail. Attach a GPU runtime.")
        
        # Download weights up-front so /progress can report a live %.
        # Wan 2.2 needs the COMPLETE model (both transformer experts); smaller
        # models stream through the same snapshot call so their shards report
        # progress too (previously they downloaded silently inside
        # from_pretrained, stuck on "downloading 0%" until READY).
        # Progress is tracked via a tqdm subclass so /ready can report live %.
        try:
            from huggingface_hub import snapshot_download
            import inspect as _inspect

            from tqdm import tqdm as _tqdm_base

            class _ProgressTqdm(_tqdm_base):
                def update(self, n=1):
                    super().update(n)
                    try:
                        total = self.total or 0
                        _load_progress["downloaded"] = self.n
                        _load_progress["total"] = total
                        _load_progress["pct"] = round(100 * self.n / total, 1) if total else 0.0
                        _load_progress["detail"] = str(self.desc or "")
                    except Exception:
                        pass

            if _is_wan22:
                print("Checking/downloading complete Wan 2.2 model weights (both experts)...")
            else:
                print(f"Checking/downloading {model_id} weights (with live progress)...")
            _set_phase("downloading", 0.0, model_id)
            _tqdm_kw = {"tqdm_class": _ProgressTqdm} if (
                "tqdm_class" in _inspect.signature(snapshot_download).parameters) else {}
            # Local paths skip the Hub call entirely.
            if os.path.isdir(model_id):
                model_path = model_id
            else:
                model_path = snapshot_download(repo_id=model_id, **_tqdm_kw)
            _set_phase("downloaded", 100.0, model_id)
            print(f"Model path resolved locally: {model_path}")
        except Exception as e:
            print(f"snapshot_download failed: {e}. Falling back to default loader.")
            model_path = model_id

        _debug_info["text_encoder_status"] = "loading"
        _debug_info["pipe_load_status"] = "loading"

        def _load_fp8_weight_scales_from_safetensors(model_dir: str) -> dict:
            """Scan all safetensors shards in model_dir and return a mapping of
            {layer_name: weight_scale_tensor} for every key ending in .weight_scale.

            Diffusers loads FP8 weights but does NOT register the companion scale
            tensors as module buffers. This function extracts them directly from
            the checkpoint files so the patch can apply correct per-layer dequantization.
            """
            import glob
            try:
                from safetensors import safe_open
            except ImportError:
                print("[FP8 patch] safetensors not available — cannot pre-load weight scales.")
                return {}

            scale_map: dict = {}
            shard_pattern = os.path.join(model_dir, "diffusion_pytorch_model*.safetensors")
            shards = sorted(glob.glob(shard_pattern))
            if not shards:
                print(f"[FP8 patch] No safetensors shards found in {model_dir}")
                return {}

            print(f"[FP8 patch] Scanning {len(shards)} shard(s) for weight_scale tensors...")
            for shard_path in shards:
                try:
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.endswith(".weight_scale"):
                                layer_name = key[: -len(".weight_scale")]
                                scale_map[layer_name] = f.get_tensor(key)
                except Exception as e:
                    print(f"[FP8 patch] Warning: could not read {os.path.basename(shard_path)}: {e}")

            print(f"[FP8 patch] Loaded {len(scale_map)} weight_scale entries from checkpoint.")
            return scale_map

        def _patch_fp8_linear_layers_dynamic(model, weight_scales: dict | None = None):
            """On-the-fly FP8 dequantization + cast for non-Hopper GPUs (no native FP8 compute).

            ModelOpt stores weights as W_fp8 = quantize(W_bf16 / weight_scale), so correct
            dequantization requires: W_real = W_fp8.to(bf16) * weight_scale.
            Skipping the scale makes weights ~3000x too large → pure noise output.

            weight_scales must be a dict keyed by layer name (e.g. 'blocks.3.attn1.to_q')
            mapping to a scalar float32 scale tensor, pre-loaded from the safetensors shards
            by _load_fp8_weight_scales_from_safetensors(). Diffusers does NOT register these
            as module buffers, so we cannot find them by inspecting module attributes.
            """
            import torch.nn as nn
            import torch.nn.functional as F
            patched = 0
            missing = 0
            logged_example = False
            ws_dict = weight_scales or {}
            for name, module in model.named_modules():
                if (
                    isinstance(module, nn.Linear)
                    and module.weight is not None
                    and module.weight.dtype == torch.float8_e4m3fn
                ):
                    # Look up the per-layer scale from the pre-loaded dict.
                    # The dict key matches named_modules() names exactly.
                    weight_scale = ws_dict.get(name)

                    if not logged_example:
                        if weight_scale is not None:
                            print(f"[FP8 patch] Scale confirmed: layer='{name}', "
                                  f"weight_scale={float(weight_scale):.6g} (from safetensors)")
                        else:
                            print(f"[FP8 patch] WARNING: no weight_scale in dict for '{name}' — "
                                  f"weights will NOT be dequantized (noise expected).")
                        logged_example = True

                    if weight_scale is None:
                        missing += 1

                    def _make_forward(mod, ws):
                        def _forward(inp):
                            # Cast FP8 → input dtype, then apply per-layer dequantization scale
                            w = mod.weight.to(inp.dtype)
                            if ws is not None:
                                w = w * ws.to(inp.dtype)
                            return F.linear(inp, w, mod.bias)
                        return _forward
                    module.forward = _make_forward(module, weight_scale)
                    patched += 1
            if missing:
                print(f"[FP8 patch] WARNING: {missing}/{patched} layers had no scale — "
                      f"those weights were cast without dequantization.")
            print(f"[FP8 patch] Patched {patched} FP8 Linear layers "
                  f"({patched - missing} with scale, {missing} without).")
            return patched

        # ── STEP 1: Load WanPipeline on CPU ────────────────────────────────────
        vae = None
        if _is_wan22:
            try:
                from diffusers import AutoencoderKLWan
                print("Loading AutoencoderKLWan VAE (CPU, float32)...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_path, subfolder="vae",
                    torch_dtype=torch.float32, low_cpu_mem_usage=True,
                )
            except Exception as e:
                print(f"Could not load VAE separately: {e}, using pipeline default.")

        pipe_dtype = torch.bfloat16 if _is_wan22 else torch.float16
        print(f"Loading WanPipeline on CPU (dtype={pipe_dtype})...")
        _set_phase("loading weights", 0.0, model_id)
        pipe_kwargs = {
            "text_encoder": None,
            "torch_dtype": pipe_dtype,
            "low_cpu_mem_usage": True,
        }
        # Load the full Wan 2.2 dual-expert MoE pipeline (transformer + transformer_2).
        # Both experts are required: transformer handles structure/motion (high-noise steps),
        # transformer_2 handles detail refinement/textures/colors (low-noise steps).
        if vae is not None:
            pipe_kwargs["vae"] = vae
        pipe = WanPipeline.from_pretrained(model_path, **pipe_kwargs)
        _debug_info["pipe_load_status"] = "success"
        print("✅ WanPipeline loaded on CPU.")
        _set_phase("moving to GPU", 0.0, model_id)

        # Detect GPU compute capability to decide FP8 execution path
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability(0)
            is_hopper_or_newer = compute_cap[0] >= 9  # sm_90 = Hopper (H100/H200)
            _is_hopper_or_newer = is_hopper_or_newer
            print(f"GPU compute capability: sm_{compute_cap[0]}{compute_cap[1]} "
                  f"({'Hopper/Newer — native FP8' if is_hopper_or_newer else 'Ampere/Older — BF16 fallback'})")
        else:
            is_hopper_or_newer = False

        # ── APPLY FP8 PATCH (non-Hopper only) ──────────────────────────────────
        # On RTX 4090 / Ada / Ampere there are no native FP8 tensor cores.
        # We override each FP8 Linear's forward() to cast weights on-the-fly to the
        # input dtype (BF16) before the matmul. This must happen while the model is
        # still on CPU (before enable_model_cpu_offload registers hooks) so that
        # accelerate moves the already-patched modules to GPU during inference.
        if not is_hopper_or_newer:
            # Pre-load weight scales from the safetensors checkpoints for each expert.
            # Diffusers does not register these as module buffers, so we must read them
            # directly from the checkpoint shards before calling the patch function.
            for expert_name, expert_subdir in [
                ("transformer",   "transformer"),
                ("transformer_2", "transformer_2"),
            ]:
                expert = getattr(pipe, expert_name, None)
                if expert is None:
                    print(f"[FP8 patch] {expert_name}: not present, skipping.")
                    continue
                # Resolve the local model directory for this expert's shards.
                # model_path may be a local snapshot path or a HuggingFace repo ID.
                expert_dir = os.path.join(model_path, expert_subdir) if os.path.isdir(model_path) else ""
                if expert_dir and os.path.isdir(expert_dir):
                    ws_dict = _load_fp8_weight_scales_from_safetensors(expert_dir)
                else:
                    print(f"[FP8 patch] {expert_name}: could not resolve local dir '{expert_dir}' — "
                          f"no pre-loaded scales available.")
                    ws_dict = {}
                n = _patch_fp8_linear_layers_dynamic(expert, weight_scales=ws_dict)
                print(f"[FP8 patch] {expert_name}: patched {n} layers for on-the-fly BF16 cast with correct dequant.")
        else:
            print("[FP8 patch] Hopper GPU detected — skipping patch (native FP8 execution).")

            # ── STEP 2: Determine execution strategy based on available VRAM ────────
        import gc
        gc.collect()
        total_vram_gib = 0.0
        if torch.cuda.is_available():
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU VRAM: {total_vram_gib:.1f} GiB total")

        # Strategy selection based on VRAM:
        #   >= 120 GB (H200 SXM 141 GB): GPU-direct + text encoder on GPU.
        #            Both BF16 experts (~56 GB) + text encoder (~22 GB) + VAE (~2 GB) = ~80 GB.
        #            Leaves 60+ GB headroom for activations. Maximum throughput, zero swap.
        #   60-120 GB (A100 SXM 80 GB): GPU-direct, but text encoder on CPU.
        #            Experts + VAE on GPU (~58 GB). Text encoder on CPU saves 22 GB for
        #            activations. CPU encoding is fast (~2s); no impact on generation quality.
        #   < 60 GB (A40 48 GB, A6000 48 GB, RTX 4090 24 GB): CPU offloading via accelerate.
        #            accelerate moves each submodel to CUDA only during its forward pass,
        #            then immediately back to CPU RAM, staying within the VRAM budget.
        _use_cpu_offload = torch.cuda.is_available() and total_vram_gib < 60.0
        use_cpu_offload = _use_cpu_offload

        # ── STEP 3: Load T5 text encoder ──────────────────────────────────────
        from transformers import UMT5EncoderModel
        if use_cpu_offload:
            _te_device = "cpu"
            print(f"Sub-60 GB GPU ({total_vram_gib:.1f} GiB) — using CPU offloading. Text encoder on CPU.")
        elif total_vram_gib >= 120.0:
            # H200 SXM (141 GB): enough VRAM for everything on GPU simultaneously.
            _te_device = "cuda"
            print(f"H200/Large-VRAM GPU ({total_vram_gib:.1f} GiB) — loading text encoder on GPU (BF16) for maximum speed.")
        elif total_vram_gib >= 60.0:
            # A100 SXM (80 GB): keep text encoder on CPU to give activations ~26 GB headroom.
            _te_device = "cpu"
            print(f"A100 class GPU ({total_vram_gib:.1f} GiB) — loading text encoder on CPU to reserve VRAM for diffusion activations.")
        else:
            _te_device = "cpu"
            print("No CUDA GPU — text encoder on CPU.")

        if not use_cpu_offload:
            print(f"Loading UMT5-XXL text_encoder in bfloat16 on {_te_device.upper()}...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=_te_device,
            )
            _debug_info["text_encoder_status"] = "success"
            _debug_info["text_encoder_device"] = _te_device
            print(f"✅ Text encoder loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

            # Attach text encoder to the pipeline before offload registration.
            pipe.text_encoder = text_encoder
            if not use_cpu_offload and _te_device == "cpu":
                # On large-VRAM GPU path with CPU text encoder: stub out device-cast methods
                # so the pipeline cannot accidentally move it to GPU mid-inference and OOM.
                text_encoder.to = lambda *a, **kw: text_encoder
                text_encoder.half = lambda *a, **kw: text_encoder
                text_encoder.float = lambda *a, **kw: text_encoder
                text_encoder.bfloat16 = lambda *a, **kw: text_encoder
        else:
            print("CPU offloading active: deferring text encoder loading to inference time to save CPU RAM.")
            _debug_info["text_encoder_status"] = "deferred"
            _debug_info["text_encoder_device"] = "deferred"

        # ── STEP 4: Move models to GPU (or enable CPU offloading) ───────────────
        # Wrap VAE decode to allow fallback attention kernels (since FlashAttention
        # doesn't support all VAE attention head dimensions).
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            _sdpa_ctx = lambda: sdpa_kernel(_sdpa_backends)
        except ImportError:
            _sdpa_ctx = lambda: torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        try:
            original_vae_decode = pipe.vae.decode
            def custom_vae_decode(*args, **kwargs):
                with _sdpa_ctx():
                    return original_vae_decode(*args, **kwargs)
            pipe.vae.decode = custom_vae_decode
            print("✅ Wrapped VAE decode to allow attention backends fallback.")
        except Exception as e:
            print(f"Could not wrap VAE decode: {e}")

        if use_cpu_offload:
            # A40 / RTX 4090 path: accelerate manages all submodels automatically.
            # enable_model_cpu_offload() must be called AFTER text_encoder is attached.
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            print(f"✅ CPU offloading enabled (A40/RTX 4090 path). "
                  f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
        else:
            # A100 SXM / H200 path: all models live on GPU permanently for max throughput.
            # 80 GB VRAM comfortably holds both BF16 experts (~56 GB) + text encoder
            # (~10 GB) + VAE (~1 GB) with ~13 GB headroom for activations.
            try:
                torch.cuda.empty_cache()
                pipe.transformer.to("cuda")
                if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
                    pipe.transformer_2.to("cuda")
                    print(f"✅ Both transformer experts on GPU. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
                else:
                    print(f"✅ Transformer on GPU. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
                pipe.vae.to("cuda")
                print(f"✅ VAE on GPU. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
                pipe.vae.decode = custom_vae_decode
                print("✅ Wrapped VAE decode to allow attention backends fallback.")
            except Exception as e:
                print(f"Could not move models to GPU: {e}")

        print(f"✅ Transformer and VAE on GPU. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

        # Scheduler shift is set dynamically per-job in _run_generation_job() based
        # on the actual generation resolution (3.0 for 480p, 5.0 for 720p).
        # It cannot be set here because the resolution may differ between jobs.
        # NOTE: pipe.scheduler.config is a FrozenDict — it must be re-instantiated,
        # not mutated in-place (the old flow_shift=5.0 assignment silently failed).

        if not is_hopper_or_newer:
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        try:
            pipe.vae.enable_tiling()
            print("✅ VAE tiling enabled.")
        except Exception as e:
            print(f"⚠️ Failed to enable VAE tiling: {e}")
        if use_cpu_offload:
            try:
                pipe.enable_attention_slicing()
                print("✅ Attention slicing enabled (low-VRAM fallback).")
            except Exception:
                pass
        else:
            try:
                pipe.disable_attention_slicing()
                print("✅ Attention slicing disabled (large-VRAM parallel path).")
            except Exception:
                pass

        # Pre-warm CUDA kernels on any large-VRAM GPU-direct path (H200, A100, etc.)
        # This compiles Triton/CUDA JIT kernels during model load so the FIRST user clip
        # runs at full speed rather than taking 3x longer for JIT compilation.
        if _is_wan22 and not use_cpu_offload and torch.cuda.is_available():
            try:
                arch_tag = f"sm_{compute_cap[0]}{compute_cap[1]}" if torch.cuda.is_available() else "cpu"
                print(f"Pre-warming CUDA kernels ({arch_tag}) to avoid first-clip JIT delay...")
                # Warm the kernels at production resolution: 5s @ 16fps @ 832x480
                _dummy_latent = torch.zeros(
                    1, 16, 21, DEFAULT_GENERATION_HEIGHT // 16, DEFAULT_GENERATION_WIDTH // 16,
                    dtype=torch.bfloat16, device="cuda"
                )
                _dummy_embeds = torch.zeros(1, 4, 4096, dtype=torch.bfloat16, device="cuda")
                with torch.no_grad():
                    # Warm up high-noise expert (timestep >= 300)
                    pipe.transformer(
                        hidden_states=_dummy_latent,
                        timestep=torch.tensor([500], device="cuda"),
                        encoder_hidden_states=_dummy_embeds,
                        return_dict=False,
                    )
                    # Warm up low-noise expert if present (timestep < 300)
                    has_t2 = hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None
                    if has_t2:
                        pipe.transformer_2(
                            hidden_states=_dummy_latent,
                            timestep=torch.tensor([100], device="cuda"),
                            encoder_hidden_states=_dummy_embeds,
                            return_dict=False,
                        )
                torch.cuda.synchronize()
                del _dummy_latent, _dummy_embeds
                torch.cuda.empty_cache()
                print(f"✅ Transformer CUDA warmup complete ({arch_tag}).")
            except Exception as _warm_err:
                print(f"Transformer CUDA warmup skipped (non-fatal): {_warm_err}")

            # ── Text Encoder warmup ───────────────────────────────────────────────
            # UMT5-XXL JIT-compiles its own attention kernels on the FIRST real
            # encode_prompt() call, adding ~3 minutes to Clip 0's wall time.
            # Running one dummy encode here burns that cost during model load
            # so ALL clips (including Clip 0) experience only ~12s text encoding.
            try:
                print("Pre-warming text encoder (UMT5-XXL) to eliminate first-clip encoding delay...")
                with torch.no_grad():
                    _dummy_prompt = "cinematic action scene shot on 35mm film"
                    pipe.encode_prompt(
                        prompt=_dummy_prompt,
                        negative_prompt="cgi, 3d render",
                        do_classifier_free_guidance=True,
                        num_videos_per_prompt=1,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("✅ Text encoder warmup complete. First clip will encode at full speed.")
            except Exception as _te_warm_err:
                print(f"Text encoder warmup skipped (non-fatal): {_te_warm_err}")


        _model_loaded = True
        _debug_info["status"] = "success"
        _set_phase("ready", 100.0, model_id)
        print("✅ Model loaded successfully! Ready to generate videos.")
    except Exception as exc:
        _model_loading = False
        _debug_info["status"] = "failed"
        _debug_info["pipe_load_status"] = "failed"
        _debug_info["pipe_load_error"] = str(exc)
        print(f"❌ Model loading failed: {exc}")
        raise


def _run_generation_job(
    job_id: str,
    prompt: str,
    duration: int,
    num_steps: int = 20,
    fps: int = DEFAULT_GENERATION_FPS,
    width: int | None = None,
    height: int | None = None,
    guidance_scale: float = 5.0,
    negative_prompt: str | None = None,
) -> None:
    """Runs in a background thread. Updates _jobs[job_id] when done."""
    import gc
    global pipe, _is_wan22, _te_device, _use_cpu_offload, total_vram_gib
    with _generation_lock:
        _jobs[job_id]["status"] = "running"
        # If Foley ran last, switch the Wan pipe back on first (no-op otherwise).
        _ensure_video()
        num_frames = max(1, int(duration * fps))
        print(f"[job {job_id[:8]}] Generating: '{prompt}' | {duration}s | {num_frames} frames @ {fps}fps")
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[job {job_id[:8]}] VRAM before generation: "
                      f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, "
                      f"{torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved")

            with torch.no_grad():
                print(f"[job {job_id[:8]}] Phase 1: Encoding prompt with negative prompt filtering...")
                if not negative_prompt:
                    negative_prompt = "cgi, 3d render, video game, anime, cartoon, sketch, painting, drawing, unreal engine, blender, smooth surfaces, low quality"
                # Combine the custom negative prompt targeting graphics with standard artifacts/noise filters
                neg_prompt = f"{negative_prompt}, blurry, low quality, distorted, ugly, oversaturated, artifact, static, flickering, glitch, deformed, noise, dull, grainy"
                
                # If CPU offloading is active (A40/RTX 4090), load text encoder dynamically
                # per-generation and free it immediately after encoding to reclaim VRAM.
                # On A100 SXM / H200 (>= 60 GB): text encoder is pre-loaded and stays on GPU.
                # On A40 (48 GB): BF16 text encoder fits transiently; loaded per-clip.
                # On RTX 4090 (24 GB): 4-bit NF4 to prevent OOM during VAE decode.
                if _use_cpu_offload:
                    from transformers import UMT5EncoderModel, BitsAndBytesConfig
                    import os
                    model_id = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
                    _large_vram = total_vram_gib >= 40.0  # A40/A6000 can afford BF16
                    if _large_vram:
                        print(f"[job {job_id[:8]}] CPU offloading: loading text encoder (BF16) on GPU...")
                        dynamic_te = UMT5EncoderModel.from_pretrained(
                            model_id,
                            subfolder="text_encoder",
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            device_map="cuda",
                        )
                    else:
                        print(f"[job {job_id[:8]}] CPU offloading: loading text encoder (4-bit NF4) on GPU...")
                        _bnb_cfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        dynamic_te = UMT5EncoderModel.from_pretrained(
                            model_id,
                            subfolder="text_encoder",
                            quantization_config=_bnb_cfg,
                            low_cpu_mem_usage=True,
                            device_map="cuda",
                        )
                    pipe.text_encoder = dynamic_te
                    _encode_device = "cuda"
                else:
                    _encode_device = pipe.text_encoder.device if (hasattr(pipe, "text_encoder") and pipe.text_encoder is not None) else ("cuda" if torch.cuda.is_available() else "cpu")

                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    device=_encode_device,
                )
                print(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

                # If text encoder was loaded dynamically, delete and free it immediately
                if _use_cpu_offload:
                    print(f"[job {job_id[:8]}] Freeing dynamically loaded text encoder to release RAM and VRAM...")
                    pipe.text_encoder = None
                    del dynamic_te
                    gc.collect()
                    torch.cuda.empty_cache()

                if width is None or height is None:
                    # Always default to 1280×720. The client (local.py) reads resolution
                    # from settings.yaml and sends it explicitly. This fallback only
                    # triggers if a raw /generate call omits width/height.
                    width, height = DEFAULT_GENERATION_WIDTH, DEFAULT_GENERATION_HEIGHT
                width = max(256, int(width))
                height = max(256, int(height))
                print(f"[job {job_id[:8]}] Diffusion loop: {num_steps} steps, res={width}x{height}, fps={fps}, guidance={guidance_scale}")

                # Move embeds to GPU and free local refs before the diffusion pass
                # so the allocator can reuse that address space.
                prompt_embeds = prompt_embeds.to("cuda")
                negative_prompt_embeds = negative_prompt_embeds.to("cuda")
                gc.collect()
                torch.cuda.empty_cache()

                # Set the correct scheduler shift for the generation resolution.
                # shift=3.0 for 480p, shift=5.0 for 720p — this controls the noise
                # schedule geometry. Wrong shift = blurry/grainy output.
                # pipe.scheduler.config is a FrozenDict so we re-instantiate the
                # scheduler instead of mutating the config in-place.
                if _is_wan22 and hasattr(pipe, "scheduler"):
                    try:
                        target_shift = 3.0 if max(width, height) < 1000 else 5.0
                        current_shift = pipe.scheduler.config.get("shift", None)
                        if current_shift != target_shift:
                            new_cfg = dict(pipe.scheduler.config)
                            new_cfg["shift"] = target_shift
                            pipe.scheduler = pipe.scheduler.__class__.from_config(new_cfg)
                            print(f"[job {job_id[:8]}] ✅ Scheduler shift → {target_shift} "
                                  f"(resolution {width}×{height})")
                    except Exception as _se:
                        print(f"[job {job_id[:8]}] ⚠️  Could not set scheduler shift: {_se}")

                video = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                ).frames[0]

                # Free embedding tensors immediately after use
                del prompt_embeds, negative_prompt_embeds
                gc.collect()
                torch.cuda.empty_cache()

            out_path = f"output_{job_id[:8]}.mp4"
            export_to_video(video, out_path, fps=fps)
            actual_duration = num_frames / fps
            print(f"[job {job_id[:8]}] ✅ Done! {num_frames} frames ({actual_duration:.1f}s) -> {out_path}")
            _jobs[job_id]["video_path"] = out_path
            _jobs[job_id]["status"] = "done"
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _debug_info["last_generation_error"] = tb
            print(tb)
            print(f"[job {job_id[:8]}] ❌ Failed: {exc}")
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["status"] = "failed"
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[job {job_id[:8]}] Post-job VRAM: "
                      f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, "
                      f"{torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_loaded,
            "load_progress": dict(_load_progress)}


def _progress_text() -> str:
    """Human-readable load status, e.g. 'downloading MODEL (42%)'."""
    phase = _load_progress.get("phase", "loading")
    pct = _load_progress.get("pct", 0.0)
    detail = _load_progress.get("detail", "")
    if phase == "downloading" and pct:
        name = detail.split("/")[-1] if detail else "weights"
        return f"downloading {name} ({pct}%)"
    if detail:
        return f"{phase}: {detail}"
    return phase


@app.get("/progress")
def progress():
    """Live model-load progress for dashboards (phase, %, current file)."""
    return {"status": "loading" if not _model_loaded else "ready",
            "device": device, "progress": dict(_load_progress),
            "text": _progress_text()}


@app.get("/ready")
def ready():
    """Check if model is loaded. If not started yet, starts loading in the background."""
    global _model_loaded, _model_loading
    if not _model_loaded:
        if _debug_info.get("status") == "failed":
            err_detail = _debug_info.get("pipe_load_error", "Unknown initialization error")
            raise HTTPException(
                status_code=500,
                detail=f"Model loading failed on GPU server: {err_detail}"
            )
        if not _model_loading:
            import threading
            threading.Thread(target=load_model, daemon=True).start()
        raise HTTPException(
            status_code=503,
            detail=f"Model {_progress_text()} in the background. "
                   "Poll /progress for live download %."
        )
    return {"status": "ready", "device": device}


@app.get("/debug")
def debug():
    """Retrieve detailed setup and loading logs/versions for remote diagnostics."""
    return _debug_info


@app.get("/vram")
def vram():
    """Report GPU memory usage and text encoder quantization status."""
    import torch
    result = {}
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved  = torch.cuda.memory_reserved()  / (1024**3)
        result["gpu_allocated_gib"] = round(allocated, 2)
        result["gpu_reserved_gib"]  = round(reserved, 2)
    if pipe is not None and hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        is_quantized = getattr(te, "is_quantized", False) or any(
            hasattr(m, "weight") and hasattr(m.weight, "quant_type")
            for m in te.modules()
        )
        result["text_encoder_is_quantized"] = is_quantized
        result["text_encoder_dtype"] = str(getattr(te, "dtype", "unknown"))

        # Calculate actual bytes consumed by text encoder parameters
        total_bytes = 0
        for p in te.parameters():
            total_bytes += p.nelement() * p.element_size()
        result["text_encoder_param_gib"] = round(total_bytes / (1024**3), 3)

    return result


@app.post("/foley")
def make_foley(video: UploadFile = File(...), prompt: str = Form(""),
               negative_prompt: str = Form(""),
               duration: float = Form(6.0), steps: int = Form(25),
               cfg_strength: float = Form(4.5), seed: int = Form(42)):
    """Self-hosted video-to-audio Foley (free, no Replicate needed).

    Upload a clip; the server switches the Foley model on (freeing the Wan
    video pipe first so it fits on small GPUs), synthesizes audio synced to
    the visuals, and returns a WAV file. Synchronous — generations take
    ~1-3 min on a T4, well within tunnel timeouts.
    """
    import gc
    import tempfile
    import torchaudio
    from mmaudio.eval_utils import generate, load_video
    from mmaudio.model.flow_matching import FlowMatching
    with _generation_lock:
        foley = _ensure_foley()
        net, fu, model = foley["net"], foley["feature_utils"], foley["model"]
        duration = max(1.0, min(float(duration), 30.0))
        steps = max(1, min(int(steps), 100))
        tmpdir = tempfile.mkdtemp(prefix="foley_")
        in_path = os.path.join(tmpdir, "clip.mp4")
        with open(in_path, "wb") as fh:
            fh.write(video.file.read())
        print(f"[foley] {video.filename or 'clip'} ({duration:.1f}s, {steps} steps, "
              f"prompt={prompt[:60]!r})")
        try:
            video_info = load_video(in_path, duration)
            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)
            duration = video_info.duration_sec
            rng = torch.Generator(device=device)
            rng.manual_seed(int(seed))
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
            seq_cfg = model.seq_cfg
            seq_cfg.duration = duration
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len,
                                   seq_cfg.sync_seq_len)
            with torch.no_grad():
                audios = generate(
                    clip_frames, sync_frames, [prompt or "cinematic sound effects"],
                    negative_text=[negative_prompt or ""],
                    feature_utils=fu, net=net, fm=fm, rng=rng,
                    cfg_strength=float(cfg_strength))
            out_wav = os.path.join(tmpdir, "foley.wav")
            torchaudio.save(out_wav, audios.float().cpu()[0], seq_cfg.sampling_rate)
            actual = video_info.duration_sec
            print(f"[foley] done! {actual:.1f}s audio -> {out_wav}")
            return FileResponse(out_wav, media_type="audio/wav", filename="foley.wav")
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            raise HTTPException(status_code=500, detail=f"Foley generation failed: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@app.post("/generate")
def start_generate(payload: dict):
    """Start a video generation job. Returns job_id immediately — use
    /status/{job_id} to poll and /result/{job_id} to download the video.
    This avoids holding the tunnel connection open for 10-15 minutes.

    Optional payload fields:
      - duration (int): requested clip length in seconds. Clamped to 1-12.
      - num_steps (int): denoising steps. Default 20 (full quality ~3 min).
        Lower values trade quality for speed. Range: 1–40.
      - fps (int): output frame rate. Defaults to 16; clamped to 8-30.
      - width (int), height (int): output resolution. Defaults to 1280x720 on
        Hopper-class GPUs and 832x480 on older fallback cards.
    """
    prompt = payload.get("prompt")
    duration = max(1, min(int(payload.get("duration", 5)), 12))
    num_steps = int(payload.get("num_steps", 20))
    num_steps = max(1, min(num_steps, 40))  # clamp to safe range
    fps = max(8, min(int(payload.get("fps", DEFAULT_GENERATION_FPS)), 30))
    guidance_scale = float(payload.get("guidance_scale", 5.0))  # Wan 2.2 optimal: 5.0–6.0
    guidance_scale = max(1.0, min(guidance_scale, 10.0))
    width = payload.get("width")
    height = payload.get("height")
    width = None if width is None else max(256, min(int(width), 1920))
    height = None if height is None else max(256, min(int(height), 1080))
    negative_prompt = payload.get("negative_prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not _model_loaded or pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not fully loaded yet. Please call /ready first.",
        )

    job_id = _uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "video_path": None,
        "error": None,
        "created_at": time.time(),
    }
    _threading.Thread(
        target=_run_generation_job,
        args=(job_id, prompt, duration, num_steps, fps, width, height, guidance_scale, negative_prompt),
        daemon=True,
    ).start()
    print(f"[job {job_id[:8]}] Queued: '{prompt}' ({duration}s, {num_steps} steps, {fps}fps, {width}x{height}, guidance={guidance_scale}, negative_prompt={negative_prompt!r})")
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Poll job status. Returns {job_id, status, error}.
    status is one of: 'queued', 'running', 'done', 'failed'.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get("/result/{job_id}")
def get_job_result(job_id: str):
    """Download the generated video once status == 'done'."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job not ready yet: {job['status']}")
    video_path = job["video_path"]
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=500, detail="Video file missing on server")
    return FileResponse(video_path, media_type="video/mp4")


@app.post("/shutdown")
def shutdown_server():
    """Release the GPU once the pipeline has everything it needs.

    A free Colab session is a limited daily allowance — idling after a run
    burns it for nothing. Refused (409) while a generation is in flight so
    a stray call can never kill a live run. Outside Colab the release step
    is skipped harmlessly (no google.colab module).
    """
    if not _generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="generation in progress")
    try:
        try:
            from google.colab import runtime as _colab_runtime
            _colab_runtime.unassign()
            return {"ok": True, "released": True,
                    "message": "Colab runtime released."}
        except Exception as exc:
            return {"ok": True, "released": False,
                    "note": f"no Colab runtime to release ({exc})"}
    finally:
        _generation_lock.release()


if __name__ == "__main__":
    # Connectivity is handled by RunPod's native HTTP proxy
    # (https://{pod_id}-8000.proxy.runpod.net) — no tunnels needed.
    # Host/port come from the environment so they stay in sync with
    # settings.yaml runpod.proxy_port without code edits.
    _host = os.environ.get("GPU_HOST", "0.0.0.0")
    _port = int(os.environ.get("GPU_PORT", "8000"))
    # GPU_WS=none serves plain HTTP without the websockets protocol: on
    # Colab, a websockets version clash raises inside the server thread where
    # nothing prints it, and the only symptom is a dead port. Default "auto"
    # keeps every other deployment exactly as before.
    _ws = os.environ.get("GPU_WS", "auto")
    print(f"Starting GPU server on {_host}:{_port} (ws={_ws})...")
    import threading
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=_host, port=_port, log_level="info", ws=_ws),
        daemon=True
    )
    server_thread.start()
    print("Uvicorn FastAPI server started.")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
