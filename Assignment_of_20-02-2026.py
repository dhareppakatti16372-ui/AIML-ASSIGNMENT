# ============================================
#           NUMPY SPEED TEST
#   Python Lists vs NumPy Arrays — 1M Numbers
# ============================================

import time
import numpy as np

SIZE = 1_000_000  # 1 Million numbers

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def timer(fn):
    """Runs a function and returns (result, elapsed_ms)."""
    start  = time.perf_counter()
    result = fn()
    end    = time.perf_counter()
    return result, (end - start) * 1000          # ms

def speedup(list_ms, numpy_ms):
    return list_ms / numpy_ms

def bar(ms, max_ms, width=30):
    """ASCII progress bar proportional to time taken."""
    filled = int((ms / max_ms) * width)
    return "█" * filled + "░" * (width - filled)

# ─────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────

def run_tests(py_list, np_array):
    results = []

    # 1. SUM
    _, t_list  = timer(lambda: sum(py_list))
    _, t_numpy = timer(lambda: np.sum(np_array))
    results.append(("Sum of all elements",       t_list, t_numpy))

    # 2. MEAN
    _, t_list  = timer(lambda: sum(py_list) / len(py_list))
    _, t_numpy = timer(lambda: np.mean(np_array))
    results.append(("Mean (average)",            t_list, t_numpy))

    # 3. MAX
    _, t_list  = timer(lambda: max(py_list))
    _, t_numpy = timer(lambda: np.max(np_array))
    results.append(("Maximum value",             t_list, t_numpy))

    # 4. MIN
    _, t_list  = timer(lambda: min(py_list))
    _, t_numpy = timer(lambda: np.min(np_array))
    results.append(("Minimum value",             t_list, t_numpy))

    # 5. ELEMENT-WISE MULTIPLY BY 2
    _, t_list  = timer(lambda: [x * 2 for x in py_list])
    _, t_numpy = timer(lambda: np_array * 2)
    results.append(("Element-wise × 2",          t_list, t_numpy))

    # 6. STANDARD DEVIATION
    mean       = sum(py_list) / len(py_list)
    _, t_list  = timer(lambda: (sum((x - mean) ** 2 for x in py_list) / len(py_list)) ** 0.5)
    _, t_numpy = timer(lambda: np.std(np_array))
    results.append(("Standard Deviation",        t_list, t_numpy))

    # 7. SORT
    _, t_list  = timer(lambda: sorted(py_list))
    _, t_numpy = timer(lambda: np.sort(np_array))
    results.append(("Sort (ascending)",          t_list, t_numpy))

    return results

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("         NUMPY SPEED TEST  —  1,000,000 Numbers")
    print("=" * 62)

    # --- Setup ---
    print(f"\n  ⚙️  Generating {SIZE:,} random numbers ...", end=" ", flush=True)
    _, t_list_create  = timer(lambda: list(range(SIZE)))
    py_list           = list(range(SIZE))
    _, t_numpy_create = timer(lambda: np.arange(SIZE))
    np_array          = np.arange(SIZE)
    print("Done!")

    print(f"\n  📦 Creation time — List : {t_list_create:>8.2f} ms")
    print(f"  📦 Creation time — NumPy: {t_numpy_create:>8.2f} ms")
    print(f"  ⚡ Speedup (creation)   : {speedup(t_list_create, t_numpy_create):>8.1f}×")

    # --- Run Tests ---
    results = run_tests(py_list, np_array)

    # --- Print Results Table ---
    print("\n" + "=" * 62)
    print(f"  {'Operation':<28}  {'List(ms)':>9}  {'NumPy(ms)':>9}  {'Speedup':>7}")
    print("-" * 62)

    max_time = max(max(r[1], r[2]) for r in results)

    for name, t_list, t_numpy in results:
        sx = speedup(t_list, t_numpy)
        print(f"  {name:<28}  {t_list:>9.2f}  {t_numpy:>9.2f}  {sx:>6.1f}×")

    print("=" * 62)

    # --- Visual Bar Chart ---
    print("\n  📊 VISUAL COMPARISON (bar = time taken)\n")
    for name, t_list, t_numpy in results:
        print(f"  {name}")
        print(f"    List  [{bar(t_list,  max_time)}] {t_list:>7.2f} ms")
        print(f"    NumPy [{bar(t_numpy, max_time)}] {t_numpy:>7.2f} ms")
        print()

    # --- Observations ---
    avg_speedup = sum(speedup(r[1], r[2]) for r in results) / len(results)

    print("=" * 62)
    print("                   OBSERVATIONS")
    print("=" * 62)
    print(f"""
  1. ⚡ NumPy is dramatically faster — on average {avg_speedup:.0f}× faster
     than Python lists across all operations. This is because
     NumPy uses optimised C code under the hood instead of
     Python's interpreted loops.

  2. 🔢 Vectorised operations (e.g. array * 2) in NumPy replace
     slow Python for-loops entirely. A list comprehension
     touches each element one by one; NumPy applies the
     operation to the whole block of memory at once (SIMD).

  3. 💾 NumPy arrays store data in contiguous, typed memory
     (e.g. int64). Python lists store pointers to individual
     objects scattered in memory, causing cache misses and
     extra overhead — making NumPy more memory-efficient too.
""")
    print("=" * 62)

if __name__ == "__main__":
    main()