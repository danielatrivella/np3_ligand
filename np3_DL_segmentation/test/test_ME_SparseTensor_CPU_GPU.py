import torch
import MinkowskiEngine as ME

device = torch.device("cpu")

# Example coordinates and features (initially on CPU)
coords_cpu = torch.randint(0, 10, size=(10, 3), dtype=torch.int64).int()
feats_cpu = torch.randn(10, 32, dtype=torch.float64).float()

# Create SparseTensor with CPU tensors
sparse_tensor_cpu = ME.SparseTensor(
    features=coords_cpu,
    coordinates=feats_cpu,
    device=device # Explicitly specify the device
)
print("CPU OK!")

#ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.CUDA)
# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Move to GPU
    coords_gpu = coords_cpu.to(device)
    feats_gpu = feats_cpu.to(device)
    # Create SparseTensor with GPU tensors
    sparse_tensor_gpu = ME.SparseTensor(
        features=feats_gpu,
        coordinates=coords_gpu,
        device=device # Explicitly specify the device
        #quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    )
    print("GPU OK!")
    print(f"SparseTensor device: {sparse_tensor_gpu.device}")
else:
    print(f"GPU not available.")