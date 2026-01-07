import torch
import numpy as np
import time
from src.model.model import MobileNetAimBot 

# 設定
CHECKPOINT = "checkpoints/2025-12-27_03-01-41/best_model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize():
    print(f"Testing on device: {DEVICE}")

    # 1. 載入模型
    model = MobileNetAimBot().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    try:
            # 1. 建立一個跟真實輸入一樣形狀的假資料 (FP32)
            # 注意：這裡要跟你的 preprocess 輸出一致
            dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
            
            # 2. 讓 Torch 跑一次模型，記錄下所有的計算路徑
            # 這會產生一個優化過的 ScriptModule
            model = torch.jit.trace(model, dummy_input)
            
            # 3. 強制優化 (Optional, 像是 Frozen graph)
            model = torch.jit.freeze(model)
            
            print("✅ TorchScript Tracing 成功！(已轉為靜態圖)")
    except Exception as e:
        print(f"⚠️ Tracing 失敗，回退到普通模式: {e}")
    # ★ 優化選項：如果你之前是測 FP16 (2.5ms)，這裡也要轉，不然 FP32 會慢一點
    # model = model.half() 
    
    # 2. 準備假資料 (修正：必須是 Tensor 且在 GPU 上)
    # 形狀要跟你的模型輸入一樣 (Batch=1, Channel=3, H=256, W=256)
    # 如果有用 .half()，這裡也要加 .half()
    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE) 
    # dummy_input = dummy_input.half() 

    # 3. 預熱 (Warmup) - 讓 GPU 醒過來
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    torch.cuda.synchronize() # 等待預熱完成

    # 4. 正式測速
    print("Starting Benchmark...")
    t_start = time.time()
    
    with torch.no_grad(): # 測速記得關掉梯度計算
        for _ in range(10000):
            _ = model(dummy_input)
            
    # ★ 關鍵：等待 GPU 真正算完
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