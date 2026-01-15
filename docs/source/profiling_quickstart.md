# Quick Start: Profiling Guide

## Running Training with Profiling

Simply run the training script as usual:

```bash
uv run python src/fruit_and_vegetable_disease/train.py
```

The profiler will automatically:
- Profile the first few batches (batches 2-4)
- Save traces to `./logs/profiler/`
- Print a summary table at the end

## Viewing Results in TensorBoard

1. Start TensorBoard:
```bash
uv run tensorboard --logdir=./logs/profiler
```

2. Open your browser to: http://localhost:6006/#pytorch_profiler

## What to Look For

### In the Console Output
- **High CPU time operations**: Candidates for optimization
- **Memory intensive operations**: Potential OOM causes

### In TensorBoard
- **Operator View**: See which operations take the most time
- **Trace View**: Visualize operation timeline
- **Memory View**: Track memory allocation patterns

## Memory-Efficient Design

The profiler is configured to use minimal memory:
- Only profiles 5 batches total (wait=1, warmup=1, active=3)
- Runs only once (repeat=1)
- No stack traces collected (with_stack=False)

This ensures profiling won't cause out-of-memory errors even on limited hardware.

## For More Details

See [profiling.md](profiling.md) for comprehensive documentation.
