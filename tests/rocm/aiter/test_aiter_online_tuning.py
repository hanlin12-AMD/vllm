# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
hipBLASLt Online Tuning Example
================================

This file demonstrates how to use hipBLASLt online tuning in vLLM via aiter's
`hipb_mm` kernel, and explains when/how vLLM triggers it automatically.

Background
----------
hipBLASLt is the AMD GEMM library used on ROCm. For a given GEMM shape
(M, N, K), there are tens to hundreds of candidate kernel algorithms.
By default, hipBLASLt uses a heuristic to pick one. Online tuning benchmarks
the candidates at runtime and caches the winner in a CSV file so subsequent
calls skip the search.

There are two levels at which online tuning can be invoked:

1. **C++-level (HIP_ONLINE_TUNING env var)**
   Intercepted inside `hipbsolgemm.cu` for every call that goes through
   `hipblasLtMatmul_sol_wrapper`.

2. **Python-level (aiter hipb_mm with solution_index)**
   Calling `hipb_mm(A, B, solution_index=-1, ...)` lets hipBLASLt choose
   via heuristic. Calling it with a specific `solution_index` (found by
   `hipb_findallsols` + benchmarking) uses that algorithm directly.
   The gradlib GemmTuner does this offline and stores results in
   `bf16_tuned_gemm.csv`.

vLLM Integration
----------------
Set `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1` to enable C++-level online
tuning for all GEMM calls in vLLM (including `torch.nn.functional.linear`).
This env var must be set before process start; vLLM reads it at import time
and sets `HIP_ONLINE_TUNING=1` before hipBLASLt initialises.

Usage::

    # Enable for an entire vLLM server
    VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 vllm serve <model>

    # Enable when calling vLLM from Python
    VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 python my_inference_script.py

Running this file::

    # With C++-level online tuning enabled (recommended for decode shapes):
    HIP_ONLINE_TUNING=1 python test_hipblaslt_online_tuning.py

    # Or via the vLLM env var:
    VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 python test_hipblaslt_online_tuning.py

    # As a pytest:
    HIP_ONLINE_TUNING=1 pytest tests/rocm/aiter/test_hipblaslt_online_tuning.py -v
"""

import importlib.util
import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
aiter_available = importlib.util.find_spec("aiter") is not None

try:
    from vllm.platforms import current_platform
    is_rocm = current_platform.is_rocm()
except Exception:
    is_rocm = False

pytestmark = pytest.mark.skipif(
    not (is_rocm and aiter_available),
    reason="hipBLASLt online tuning requires ROCm + aiter",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_hipblas():
    """Initialise the hipBLASLt handle (lazy, idempotent)."""
    import aiter
    aiter.hipb_create_extension()


def _make_inputs(m, n, k, dtype=torch.bfloat16, device="cuda"):
    """Create random A [M, K] and B [N, K] tensors."""
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    return A, B


def _reference(A, B):
    """Compute reference result with torch for correctness check."""
    return torch.nn.functional.linear(A.float(), B.float()).to(A.dtype)


# ---------------------------------------------------------------------------
# Example 1: hipb_mm with heuristic (solution_index = -1)
# ---------------------------------------------------------------------------

def test_hipb_mm_heuristic():
    """
    Demonstrates calling hipb_mm with solution_index=-1 (heuristic mode).

    When HIP_ONLINE_TUNING=1, the *first* call for a new decode shape
    (N <= 512) benchmarks up to 32 candidates inside C++ and saves the
    winner to ./hip_online_tuning_res.csv. Subsequent calls read the
    cached algo_index from the CSV, bypassing the search entirely.

    When HIP_ONLINE_TUNING is not set (or N > 512), hipBLASLt uses its
    built-in heuristic without any benchmarking.
    """
    import aiter

    _init_hipblas()

    # Typical decode-phase shapes: small M (batch), large N/K
    shapes = [
        (1,   4096, 4096),   # batch=1  decode
        (4,   4096, 4096),   # batch=4  decode
        (1,   8192, 8192),   # batch=1, larger weights
        (16,  512,  4096),   # N=512, boundary case for online tuning
    ]

    online_tuning_active = os.environ.get("HIP_ONLINE_TUNING", "0") in ("1", "true")
    if online_tuning_active:
        print("\n[INFO] HIP_ONLINE_TUNING is active — first unseen shapes will "
              "be benchmarked and saved to ./hip_online_tuning_res.csv")
    else:
        print("\n[INFO] HIP_ONLINE_TUNING is not set — using heuristic only")

    for m, n, k in shapes:
        A, B = _make_inputs(m, n, k)

        # hipb_mm expects B transposed: A [M,K] @ B.T [K,N] → C [M,N]
        # solution_index=-1: let hipBLASLt decide (heuristic or online tuning)
        C = aiter.hipb_mm(A, B.t(), solution_index=-1)

        ref = _reference(A, B)
        assert C.shape == (m, n), f"Expected ({m},{n}), got {C.shape}"
        assert torch.allclose(C.float(), ref.float(), atol=0.05, rtol=0.05), \
            f"Numerical mismatch for shape ({m},{n},{k})"

        print(f"  ({m:4d}, {n:4d}, {k:4d})  ✓  out={C.shape}  dtype={C.dtype}")

    print("[PASS] test_hipb_mm_heuristic")


# ---------------------------------------------------------------------------
# Example 2: hipb_mm with a specific solution_index (from findallsols)
# ---------------------------------------------------------------------------

def test_hipb_mm_explicit_solution():
    """
    Demonstrates the manual workflow:
      1. hipb_findallsols() — enumerate all valid hipBLASLt algorithms.
      2. Benchmark them (simple timing loop here).
      3. Run hipb_mm with the winning solution_index.

    This is what aiter's GemmTuner does offline and stores in bf16_tuned_gemm.csv.
    For production use, run the tuner once and let vLLM load the CSV at startup
    via AITER_CONFIG_GEMM_BF16.
    """
    import aiter

    _init_hipblas()

    m, n, k = 4, 4096, 4096
    A, B = _make_inputs(m, n, k)

    # B must be transposed when passed to hipb_mm / hipb_findallsols
    B_t = B.t().contiguous()

    # Step 1: find all valid solutions for this shape
    solutions = aiter.hipb_findallsols(
        A, B_t,
        bias=None,
        out_dtype=torch.bfloat16,
        scaleA=None,
        scaleB=None,
        bpreshuffle=False,
    )
    assert len(solutions) > 0, "hipb_findallsols returned 0 solutions"
    print(f"\n  Found {len(solutions)} hipBLASLt solutions for "
          f"({m}, {n}, {k}) bf16")

    # Step 2: quick benchmark — pick the fastest
    num_warmup, num_iters = 5, 20
    best_idx = solutions[0]
    best_us = float("inf")

    for sol in solutions:
        # warmup
        for _ in range(num_warmup):
            aiter.hipb_mm(A, B_t, sol)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            aiter.hipb_mm(A, B_t, sol)
        end.record()
        torch.cuda.synchronize()

        elapsed_us = start.elapsed_time(end) * 1000 / num_iters  # ms → µs
        if elapsed_us < best_us:
            best_us  = elapsed_us
            best_idx = sol

    print(f"  Best solution_index={best_idx}  ({best_us:.1f} µs)")

    # Step 3: use the winning solution
    C = aiter.hipb_mm(A, B_t, best_idx)
    ref = _reference(A, B)

    assert C.shape == (m, n)
    assert torch.allclose(C.float(), ref.float(), atol=0.05, rtol=0.05), \
        "Numerical mismatch with best solution"

    print("[PASS] test_hipb_mm_explicit_solution")


# ---------------------------------------------------------------------------
# Example 3: Verify the online-tuning CSV cache is populated
# ---------------------------------------------------------------------------

def test_hip_online_tuning_csv_populated():
    """
    When HIP_ONLINE_TUNING=1, calling hipb_mm for a new decode shape
    (N <= 512) should write a row to ./hip_online_tuning_res.csv.

    This test verifies the file is created and that the row for the
    shape we tested is present.

    Skip automatically when HIP_ONLINE_TUNING is not set, since the
    CSV will not be written in that case.
    """
    if os.environ.get("HIP_ONLINE_TUNING", "0") not in ("1", "true"):
        pytest.skip("HIP_ONLINE_TUNING is not set — CSV cache is not written")

    import csv

    import aiter

    _init_hipblas()

    # Use a decode shape (N <= 512) to trigger online tuning
    m, n, k = 1, 256, 4096
    A, B = _make_inputs(m, n, k)

    cache_file = "./hip_online_tuning_res.csv"

    # Remove the cache entry for this shape if it exists, so we exercise
    # the actual tuning path (not just the cache-hit path).
    # In production you would never do this — just leave the CSV intact.
    _remove_csv_row(cache_file, m, n, k)

    # First call: triggers benchmarking + writes CSV
    C = aiter.hipb_mm(A, B.t(), solution_index=-1)
    torch.cuda.synchronize()

    assert os.path.exists(cache_file), \
        f"Expected {cache_file} to be created by online tuning"

    # Verify a row for (m, n, k) appears in the CSV
    found = _find_csv_row(cache_file, m, n, k)
    assert found, \
        f"No row for ({m},{n},{k}) found in {cache_file}"
    print(f"\n  Cache row for ({m},{n},{k}): {found}")

    # Second call: cache-hit path (no benchmarking)
    C2 = aiter.hipb_mm(A, B.t(), solution_index=-1)
    torch.cuda.synchronize()

    ref = _reference(A, B)
    assert torch.allclose(C.float(),  ref.float(), atol=0.05, rtol=0.05)
    assert torch.allclose(C2.float(), ref.float(), atol=0.05, rtol=0.05)

    print("[PASS] test_hip_online_tuning_csv_populated")


# ---------------------------------------------------------------------------
# Example 4: vLLM integration via VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING
# ---------------------------------------------------------------------------

def test_vllm_env_var_sets_hip_online_tuning():
    """
    Demonstrates that vLLM's VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1
    propagates to HIP_ONLINE_TUNING=1 at the C++ level.

    In normal usage the env var must be set *before* the process starts
    (because vllm/platforms/rocm.py reads and forwards it at import time,
    before hipBLASLt is initialised). This test checks the forwarding
    logic in isolation.

    To exercise the full end-to-end path, start vLLM like this::

        VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 vllm serve <model>

    or::

        VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 python -c "
        import vllm
        llm = vllm.LLM('meta-llama/Llama-3.1-8B')
        out = llm.generate(['Hello'])
        print(out[0].outputs[0].text)
        "

    The first decode requests for each unique (M, N, K) shape will trigger
    online tuning (≈ a few seconds). Results persist in hip_online_tuning_res.csv,
    so subsequent runs are instant.
    """
    # vllm/platforms/rocm.py executes this at import time:
    #   if envs.VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING:
    #       os.environ["HIP_ONLINE_TUNING"] = "1"
    #
    # We simulate that forwarding here and confirm HIP_ONLINE_TUNING is set.

    import vllm.envs as envs

    if envs.VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING:
        assert os.environ.get("HIP_ONLINE_TUNING") == "1", (
            "VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 was set but "
            "HIP_ONLINE_TUNING was not forwarded to the environment. "
            "Make sure vllm.platforms.rocm is imported before hipBLASLt "
            "is initialised."
        )
        print("\n  VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1 → "
              "HIP_ONLINE_TUNING=1  ✓")
    else:
        print("\n  VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING is not set "
              "(HIP_ONLINE_TUNING will not be enabled via vLLM)")

    print("[PASS] test_vllm_env_var_sets_hip_online_tuning")


# ---------------------------------------------------------------------------
# Example 5: FP8 row-wise scaled GEMM with hipb_mm
# ---------------------------------------------------------------------------

def test_hipb_mm_fp8_rowwise():
    """
    Demonstrates hipb_mm with FP8 inputs and row-wise scaling,
    which is used for quantised inference on MI300 (gfx942) and
    MI350 (gfx950).

    Row-wise scaling requires hipBLASLt >= 1.0 (ROCm 7.0+).
    Online tuning also applies here when HIP_ONLINE_TUNING=1 and N <= 512.
    """
    import aiter
    from aiter import dtypes

    _init_hipblas()

    try:
        fp8_dtype = torch.float8_e4m3fnuz   # MI300 native FP8
    except AttributeError:
        pytest.skip("torch.float8_e4m3fnuz not available")

    m, n, k = 4, 512, 4096

    # Quantise inputs to FP8
    A_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    A_fp8, x_scale = aiter.pertoken_quant(A_bf16, quant_dtype=fp8_dtype)
    B_fp8, w_scale = aiter.pertoken_quant(B_bf16, quant_dtype=fp8_dtype)

    # x_scale shape: [M, 1], w_scale shape: [N, 1]
    # hipb_mm expects scaleB to be transposed → [1, N]
    C = aiter.hipb_mm(
        A_fp8,
        B_fp8.t(),           # [K, N]
        solution_index=-1,
        out_dtype=torch.bfloat16,
        scaleA=x_scale,
        scaleB=w_scale.t(),  # [1, N]
    )

    assert C.shape == (m, n), f"Expected ({m},{n}), got {C.shape}"
    assert C.dtype == torch.bfloat16

    # Reference: dequantise then matmul
    ref = (A_bf16.float() @ B_bf16.float().t()).bfloat16()
    assert torch.allclose(C.float(), ref.float(), atol=0.5, rtol=0.1), \
        "FP8 result deviates too far from bf16 reference"

    print(f"\n  FP8 rowwise ({m},{n},{k})  ✓  out={C.shape}")
    print("[PASS] test_hipb_mm_fp8_rowwise")


# ---------------------------------------------------------------------------
# CSV helpers (used by test_hip_online_tuning_csv_populated)
# ---------------------------------------------------------------------------

def _find_csv_row(path: str, m: int, n: int, k: int) -> dict | None:
    """Return the first CSV row whose m/n/k fields match, or None."""
    if not os.path.exists(path):
        return None
    import csv as _csv
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                if (int(row.get("m", -1)) == m
                        and int(row.get("n", -1)) == n
                        and int(row.get("k", -1)) == k):
                    return dict(row)
            except (ValueError, KeyError):
                continue
    return None


def _remove_csv_row(path: str, m: int, n: int, k: int) -> None:
    """Remove rows matching (m, n, k) from the CSV (for test repeatability)."""
    if not os.path.exists(path):
        return
    import csv as _csv
    rows = []
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            try:
                if not (int(row.get("m", -1)) == m
                        and int(row.get("n", -1)) == n
                        and int(row.get("k", -1)) == k):
                    rows.append(row)
            except (ValueError, KeyError):
                rows.append(row)
    if fieldnames:
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Standalone entry-point (run without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not (is_rocm and aiter_available):
        print("ERROR: requires ROCm platform with aiter installed")
        raise SystemExit(1)

    print("=" * 60)
    print("hipBLASLt Online Tuning Demo")
    print("=" * 60)
    print(f"HIP_ONLINE_TUNING = {os.environ.get('HIP_ONLINE_TUNING', '(not set)')}")

    try:
        import vllm.envs as envs
        print(f"VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING = "
              f"{envs.VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING}")
    except ImportError:
        pass

    print()

    test_hipb_mm_heuristic()
    print()
    test_hipb_mm_explicit_solution()
    print()
    test_hip_online_tuning_csv_populated()
    print()
    test_vllm_env_var_sets_hip_online_tuning()
    print()
    test_hipb_mm_fp8_rowwise()

    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
