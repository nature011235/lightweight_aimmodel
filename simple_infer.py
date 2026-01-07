import torch
import numpy as np
import time
from src.model.model import MobileNetAimBot 


CHECKPOINT = "checkpoints/2025-12-27_03-01-41/best_model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize():
    print(f"Testing on device: {DEVICE}")

    model = MobileNetAimBot().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    try:
 
            dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
            

            model = torch.jit.trace(model, dummy_input)
            
    
            model = torch.jit.freeze(model)
            
            print("✅ TorchScript Tracing 成功！(已轉為靜態圖)")
    except Exception as e:
        print(f"⚠️ Tracing 失敗，回退到普通模式: {e}")

    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE) 
    # dummy_input = dummy_input.half() 

  
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    torch.cuda.synchronize() 


    print("Starting Benchmark...")
    t_start = time.time()
    
    with torch.no_grad(): 
        for _ in range(10000):
            _ = model(dummy_input)
            
   
    torch.cuda.synchronize()
    t_end = time.time()

    avg_time = (t_end - t_start) / 10000 * 1000
    fps = 1000 / avg_time
    
    print(f"=======================================")
    print(f"Our Model (MobileNet) Inference Time: {avg_time:.2f} ms")
    print(f"Equivalent FPS: {fps:.1f}")
    print(f"=======================================")

if __name__ == "__main__":
    visualize()