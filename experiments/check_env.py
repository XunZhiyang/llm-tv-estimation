#!/usr/bin/env python3
"""
Environment checker for TV distance estimation experiments.
Run this first before anything else.

Usage:
    python3 experiments/check_env.py
"""

import sys
import os
import shutil


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def ok(msg):   print(f"  [ OK ] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")


def check_package(display_name, import_name=None, min_version=None, install_name=None,
                  optional=False):
    import_name = import_name or display_name
    install_name = install_name or display_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        if min_version:
            try:
                from packaging.version import Version
                if Version(version) < Version(min_version):
                    warn(f"{display_name} {version} (need >= {min_version}  →  python3 -m pip install -U {install_name})")
                    return False
            except ImportError:
                pass  # can't check version without packaging
        ok(f"{display_name} {version}")
        return True
    except ImportError:
        if optional:
            warn(f"{display_name} not found (optional)  →  python3 -m pip install {install_name}")
        else:
            fail(f"{display_name} not found  →  python3 -m pip install {install_name}")
        return False


# ── Python ────────────────────────────────────────────────────────────────────
section("Python")
v = sys.version_info
if v >= (3, 9):
    ok(f"Python {sys.version.split()[0]}")
else:
    fail(f"Python {sys.version.split()[0]}  (need >= 3.9)")

# ── Required packages ─────────────────────────────────────────────────────────
section("Required packages")
has_torch          = check_package("torch",          min_version="2.1.0")
has_transformers   = check_package("transformers",   min_version="4.51.0")
has_accelerate     = check_package("accelerate",     min_version="0.30.0")
has_hf_hub         = check_package("huggingface_hub", import_name="huggingface_hub", install_name="huggingface_hub")

section("Optional packages")
check_package("numpy")
# bitsandbytes is only needed for the optional --quant_{pi,mu} int8/nf4/fp4 paths
has_bnb            = check_package("bitsandbytes",   min_version="0.41.0", optional=True)
check_package("datasets", optional=True)
check_package("tqdm", optional=True)

# ── CUDA ──────────────────────────────────────────────────────────────────────
section("CUDA / GPU")
if has_torch:
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            mem_gb = p.total_memory / 1e9
            msg = f"GPU {i}: {p.name}  {mem_gb:.1f} GB"
            if mem_gb >= 80:
                ok(msg + "  (fits two 8B models easily)")
            elif mem_gb >= 40:
                ok(msg + "  (fits two 8B models in mixed precision)")
            else:
                warn(msg + "  (may be tight for two 8B models)")
    else:
        fail("CUDA not available — check module environment")
        info("On TACC Vista, try:  module load cuda  or check $CUDA_HOME")
else:
    info("Skipping CUDA check (torch not installed)")

# ── bitsandbytes CUDA test ────────────────────────────────────────────────────
section("bitsandbytes CUDA integration")
if has_bnb and has_torch:
    try:
        import bitsandbytes as bnb
        import torch
        if torch.cuda.is_available():
            # 8-bit smoke test
            layer8 = bnb.nn.Linear8bitLt(16, 16, has_fp16_weights=False).cuda()
            x = torch.randn(2, 16, device='cuda', dtype=torch.float16)
            _ = layer8(x)
            ok("bitsandbytes 8-bit CUDA works")
            # 4-bit smoke test (covers the nf4/fp4 code path of the --quant_{pi,mu} options)
            layer4 = bnb.nn.Linear4bit(16, 16, quant_type="nf4").cuda()
            _ = layer4(x)
            ok("bitsandbytes 4-bit CUDA works")
        else:
            warn("Skipping bitsandbytes CUDA test (no GPU)")
    except Exception as e:
        fail(f"bitsandbytes CUDA test failed: {e}")
        info("Try: pip install -U bitsandbytes")
else:
    info("Skipping bitsandbytes test")

# ── Disk space ────────────────────────────────────────────────────────────────
section("Disk space")
info("Qwen3-8B bf16 ≈ 16 GB;  with int8+NF4 copies ≈ 28 GB total")

scratch = os.environ.get('SCRATCH')
home    = os.path.expanduser('~')

for label, path in [('$SCRATCH', scratch), ('$HOME', home)]:
    if path and os.path.exists(path):
        free_gb = shutil.disk_usage(path).free / 1e9
        msg = f"{label} ({path}): {free_gb:.1f} GB free"
        if free_gb >= 50:
            ok(msg)
        elif free_gb >= 25:
            warn(msg + "  (tight — recommend $SCRATCH for HF cache)")
        else:
            fail(msg + "  (insufficient)")

hf_home = os.environ.get('HF_HOME') or os.environ.get('HUGGINGFACE_HUB_CACHE')
if hf_home:
    ok(f"HF_HOME = {hf_home}")
elif scratch:
    warn("HF_HOME not set — models will go to ~/.cache/huggingface (may hit $HOME quota)")
    info(f"Fix:  export HF_HOME=$SCRATCH/.cache/huggingface")
else:
    info("HF_HOME not set")

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
missing = []
if not has_torch:        missing.append("torch")
if not has_transformers: missing.append("transformers>=4.51.0")
if not has_accelerate:   missing.append("accelerate")
if not has_hf_hub:       missing.append("huggingface_hub")
if not has_bnb:
    info("bitsandbytes not installed — only needed for the optional int8/nf4/fp4 quantization paths")

if missing:
    fail("Missing packages: " + ", ".join(missing))
    # Quote version specifiers so the command is safe to copy-paste into any shell
    quoted = [f"'{p}'" if ">=" in p or "<=" in p else p for p in missing]
    print(f"\n  Install with:\n    python3 -m pip install {' '.join(quoted)}\n")
    sys.exit(1)
else:
    ok("All dependencies present. Ready to download models and run experiments.")
    if scratch:
        print(f"\n  Recommended before downloading:\n    export HF_HOME=$SCRATCH/.cache/huggingface\n")
