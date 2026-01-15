# Profiling with PyTorch Profiler

This document explains how profiling is implemented in the training script and how to view the results.

## Overview

The training script uses PyTorch's built-in profiler to analyze performance bottlenecks. Profiling is configured to minimize memory usage by only profiling a few batches at the beginning of training.

## Implementation Details

### Profiler Configuration

The profiler is configured with a schedule that:
- **Waits** 1 batch to skip initial setup overhead
- **Warms up** for 1 batch to stabilize measurements
- **Actively profiles** 3 batches to collect data
- **Runs once** to minimize memory usage

### Memory Optimization

To prevent memory issues:
- `with_stack=False`: Disables stack traces collection
- `schedule()`: Limits profiling to only 5 batches total
- `profile_memory=True`: Tracks memory usage without excessive overhead
- `record_shapes=True`: Records tensor shapes for analysis

### Activities Profiled

- **CPU**: Always profiled
- **CUDA**: Automatically enabled if GPU is available

## Viewing Results

### Command Line Output

After training completes, a profiling summary is printed showing:
- Top 10 operations by CPU time
- Top 10 operations by memory usage (if GPU available)

### TensorBoard Visualization

Profiling traces are saved to `./logs/profiler/` directory.

To view in TensorBoard:

```bash
uv run tensorboard --logdir=./logs/profiler
```

Then open your browser to [http://localhost:6006/#pytorch_profiler](http://localhost:6006/#pytorch_profiler)

### TensorBoard Views

TensorBoard provides several views:

1. **Overview**: Shows overall performance metrics
2. **Operator View**: Detailed breakdown by operation type
3. **Kernel View**: GPU kernel execution details
4. **Trace View**: Timeline visualization of operations
5. **Memory View**: Memory allocation and usage patterns

## Interpreting Results

### Key Metrics

- **Self CPU Time**: Time spent in the operation itself (excluding child calls)
- **Total CPU Time**: Time including all child operations
- **CPU Memory**: Memory allocated during the operation
- **# of Calls**: How many times the operation was called

### Common Bottlenecks

Look for:
- Operations with high CPU time that could be optimized
- Memory-intensive operations that might cause OOM errors
- Data loading operations that could benefit from faster I/O
- Unnecessary data transfers between CPU and GPU

## Customizing Profiling

To profile more or fewer batches, modify the schedule in [train.py](../../src/fruit_and_vegetable_disease/train.py):

```python
profiler_schedule = schedule(
    wait=1,      # batches to skip
    warmup=1,    # batches to warm up
    active=3,    # batches to profile
    repeat=1     # how many times to repeat
)
```

To disable profiling entirely, comment out the profiler setup and `prof.step()` calls.
