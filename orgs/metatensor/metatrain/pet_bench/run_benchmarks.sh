#!/usr/bin/env bash
# Run PET compile + DDP benchmarks and generate plots.
#
# Usage: bash run_benchmarks.sh [--compile-only | --ddp-only | --plot-only]
#
# Must be run from the pixi workspace root:
#   cd pixi_envs/orgs/metatensor/metatrain
#   bash pet_bench/run_benchmarks.sh
#
# Prerequisites:
#   - pixi installed ($HOME/.pixi/bin/pixi)
#   - metatrain checked out at metatrain/ with petPartialCompile branch
#   - For DDP: CUDA GPU available
#   - For plots: matplotlib available via pixi

set -euo pipefail

BENCH_DIR="pet_bench"
RESULTS_DIR="$BENCH_DIR/results"
PLOTS_DIR="$BENCH_DIR/plots"
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

echo "=== PET Compile + DDP Benchmark ==="
echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo 'unknown')"
echo ""

RUN_COMPILE=true
RUN_DDP=true
RUN_PLOT=true

case "${1:-all}" in
    --compile-only) RUN_DDP=false; RUN_PLOT=false ;;
    --ddp-only)     RUN_COMPILE=false; RUN_PLOT=false ;;
    --plot-only)    RUN_COMPILE=false; RUN_DDP=false ;;
esac

if $RUN_COMPILE; then
    echo "--- Running compile benchmark (eager vs compiled, fixed + adaptive) ---"
    pixi run uv run --project metatrain --extra pet \
        python "$BENCH_DIR/bench_compile.py" \
        --steps 50 --batch-size 8 \
        --output "$RESULTS_DIR/bench_compile.json" \
        2>&1 | tee "$RESULTS_DIR/bench_compile.log"
    echo ""
fi

if $RUN_DDP; then
    echo "--- Running DDP benchmark (4 configs: eager/compiled x single/DDP) ---"
    pixi run uv run --project metatrain --extra pet \
        python "$BENCH_DIR/bench_ddp.py" \
        --steps 30 --batch-size 8 \
        --output "$RESULTS_DIR/bench_ddp.json" \
        2>&1 | tee "$RESULTS_DIR/bench_ddp.log"
    echo ""
fi

if $RUN_PLOT; then
    echo "--- Generating plots ---"
    PLOT_ARGS=""
    [ -f "$RESULTS_DIR/bench_compile.json" ] && PLOT_ARGS="$PLOT_ARGS --compile $RESULTS_DIR/bench_compile.json"
    [ -f "$RESULTS_DIR/bench_ddp.json" ] && PLOT_ARGS="$PLOT_ARGS --ddp $RESULTS_DIR/bench_ddp.json"

    if [ -n "$PLOT_ARGS" ]; then
        pixi run uv run --project metatrain --extra pet --with matplotlib \
            python "$BENCH_DIR/plot_benchmarks.py" \
            $PLOT_ARGS --output-dir "$PLOTS_DIR"
    else
        echo "No results JSON found. Run benchmarks first."
    fi
fi

echo ""
echo "=== Done ==="
echo "Results: $RESULTS_DIR/"
echo "Plots:   $PLOTS_DIR/"
