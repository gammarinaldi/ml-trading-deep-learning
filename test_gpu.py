import torch
import sys

print("=" * 60)
print("GPU & CUDA INFORMATION")
print("=" * 60)

# Basic GPU Information
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA Available: ✅ YES")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    
    # Current GPU Details
    current_device = torch.cuda.current_device()
    print(f"Current GPU Device ID: {current_device}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(current_device)}")
    
    # GPU Memory Information
    gpu_memory = torch.cuda.get_device_properties(current_device)
    total_memory = gpu_memory.total_memory / (1024**3)  # Convert to GB
    print(f"Total GPU Memory: {total_memory:.2f} GB")
    
    # Compute Capability
    major = gpu_memory.major
    minor = gpu_memory.minor
    print(f"Compute Capability: {major}.{minor}")
    
    # Multiprocessor Count
    print(f"Multiprocessor Count: {gpu_memory.multi_processor_count}")
    
    # Current Memory Usage
    allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
    cached = torch.cuda.memory_reserved(current_device) / (1024**3)
    print(f"Currently Allocated Memory: {allocated:.3f} GB")
    print(f"Currently Cached Memory: {cached:.3f} GB")
    print(f"Available Memory: {total_memory - cached:.3f} GB")
    
    # All Available GPUs
    if torch.cuda.device_count() > 1:
        print("\n" + "=" * 40)
        print("ALL AVAILABLE GPUs:")
        print("=" * 40)
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            memory_gb = properties.total_memory / (1024**3)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {memory_gb:.2f} GB")
            print(f"  Compute: {properties.major}.{properties.minor}")
            print(f"  Multiprocessors: {properties.multi_processor_count}")
            print()

else:
    print(f"CUDA Available: ❌ NO")
    print("CUDA is not available. Install PyTorch with CUDA support:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 40)
print("SYSTEM INFORMATION")
print("=" * 40)
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# Device Selection Test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Selected Device: {device}")

# Simple GPU Test
if torch.cuda.is_available():
    print("\n" + "=" * 40)
    print("GPU FUNCTIONALITY TEST")
    print("=" * 40)
    try:
        # Create a simple tensor and move to GPU
        test_tensor = torch.randn(1000, 1000).to(device)
        result = torch.mm(test_tensor, test_tensor.t())
        print("✅ GPU tensor operations: WORKING")
        print(f"Test tensor shape: {test_tensor.shape}")
        print(f"Result tensor device: {result.device}")
        
        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()
        print("✅ GPU memory cleanup: COMPLETED")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

print("\n" + "=" * 60)
print("GPU TEST COMPLETE")
print("=" * 60)