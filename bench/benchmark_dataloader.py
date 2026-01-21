import time
import torch
from pathlib import Path
from fruit_and_vegetable_disease.data import create_datasets
from fruit_and_vegetable_disease.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PROCESSED_DATA_DIR = Path("data/processed")


def benchmark_dataloader(batch_size: int = 32, num_workers: int = 0, num_batches: int = 50):
    """Benchmark data loading with different configurations.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        num_batches: Number of batches to benchmark
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking with num_workers={num_workers}, batch_size={batch_size}")
    print(f"{'='*60}")
    
    # Load dataset
    train_set, _ = create_datasets(str(PROCESSED_DATA_DIR))
    train_dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),  # Use pinned memory for GPU
    )
    
    # Load model
    model = Model(num_classes=2).to(DEVICE)
    model.train()
    
    # Timing variables
    data_loading_times = []
    model_forward_times = []
    model_backward_times = []
    
    # Warmup
    print("Warming up...")
    for i, (img, target) in enumerate(train_dataloader):
        if i >= 5:
            break
        img, target = img.to(DEVICE), target.to(DEVICE)
        output = model(img)
    
    print("Running benchmark...")
    data_start = time.time()
    
    for i, (img, target) in enumerate(train_dataloader):
        if i >= num_batches:
            break
        
        # Time data loading (includes transfer to device)
        data_end = time.time()
        if i > 0:  # Skip first iteration
            data_loading_times.append(data_end - data_start)
        
        # Move to device
        transfer_start = time.time()
        img, target = img.to(DEVICE), target.to(DEVICE)
        transfer_end = time.time()
        
        # Time model forward pass
        forward_start = time.time()
        output = model(img)
        loss = torch.nn.functional.cross_entropy(output, target)
        forward_end = time.time()
        model_forward_times.append(forward_end - forward_start)
        
        # Time backward pass
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        model_backward_times.append(backward_end - backward_start)
        
        # Start timing next data loading
        data_start = time.time()
    
    # Calculate statistics
    avg_data_loading = sum(data_loading_times) / len(data_loading_times) * 1000  # ms
    avg_forward = sum(model_forward_times) / len(model_forward_times) * 1000  # ms
    avg_backward = sum(model_backward_times) / len(model_backward_times) * 1000  # ms
    total_iter_time = avg_data_loading + avg_forward + avg_backward
    
    print(f"\nResults (averaged over {len(data_loading_times)} batches):")
    print(f"  Data Loading:     {avg_data_loading:6.2f} ms ({avg_data_loading/total_iter_time*100:.1f}%)")
    print(f"  Forward Pass:     {avg_forward:6.2f} ms ({avg_forward/total_iter_time*100:.1f}%)")
    print(f"  Backward Pass:    {avg_backward:6.2f} ms ({avg_backward/total_iter_time*100:.1f}%)")
    print(f"  Total/Iteration:  {total_iter_time:6.2f} ms")
    print(f"  Throughput:       {1000/total_iter_time*batch_size:.1f} samples/sec")
    
    return {
        "num_workers": num_workers,
        "avg_data_loading_ms": avg_data_loading,
        "avg_forward_ms": avg_forward,
        "avg_backward_ms": avg_backward,
        "total_ms": total_iter_time,
        "data_loading_percentage": avg_data_loading/total_iter_time*100,
        "throughput": 1000/total_iter_time*batch_size,
    }


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    results = []
    
    # Test different num_workers configurations
    for num_workers in [0, 2, 4, 8]:
        try:
            result = benchmark_dataloader(batch_size=32, num_workers=num_workers, num_batches=50)
            results.append(result)
        except Exception as e:
            print(f"Error with num_workers={num_workers}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':<10} {'Data Load %':<15} {'Total Time':<15} {'Throughput':<15}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['num_workers']:<10} {r['data_loading_percentage']:<15.1f} {r['total_ms']:<15.2f} {r['throughput']:<15.1f}")
    
    # Find optimal configuration
    best = min(results, key=lambda x: x['total_ms'])
    print(f"\n✓ Best configuration: num_workers={best['num_workers']}")
    print(f"  Improvement vs num_workers=0: {(1 - best['total_ms']/results[0]['total_ms'])*100:.1f}%")
