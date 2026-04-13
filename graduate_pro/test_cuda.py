# 创建测试脚本 test_gpu.py
import torch
import time

print("=== GPU测试 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 性能测试
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # GPU计算测试
    start_time = time.time()
    x = torch.randn(1000, 1000).to(device)
    y = torch.matmul(x, x)
    torch.cuda.synchronize()  # 等待GPU完成
    gpu_time = time.time() - start_time
    
    # CPU对比测试  
    start_time = time.time()
    x_cpu = torch.randn(1000, 1000)
    y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start_time
    
    print(f"GPU计算时间: {gpu_time:.4f}秒")
    print(f"CPU计算时间: {cpu_time:.4f}秒") 
    print(f"GPU加速比: {cpu_time/gpu_time:.1f}x")
else:
    print("❌ CUDA仍然不可用")