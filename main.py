import ctypes
import os
import time

import cv2
import dxcam
import mss
import psutil
import torch
import win32api
import win32con

from src.model.model import MobileNetAimBot

p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)

CHECKPOINT_PATH = "checkpoints/2025-12-27_03-01-41/best_model_weights.pth"  # 改這裡！
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOV_SIZE = 350
CONF_THRESHOLD = 0.5

# windows api
# def move_mouse(x, y):
#     win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)

# --------------------------------------------------------------
# logitech driver
try:
    gm = ctypes.CDLL(os.path.join(os.getcwd(), "ghub_device.dll"))
    gmok = gm.device_open() == 1
    if not gmok:
        print("don't have driver or ghub")
    else:
        print("sucefully init")

    def move_mouse(x, y, abs_move=False):
        if gmok:
            gm.moveR(int(x), int(y), abs_move)
        else:
            print("error")

except Exception as e:
    print(f"Can't load Logitech DLL: {e}")
    print("make sure ghub_device.dll exist")

    # Fallback to win32api (但在 Apex 裡沒用)
    def move_mouse(x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
# --------------------------------------------------------------------
# pydirect input

# pydirectinput.FAILSAFE = False
# def move_mouse(x, y):
#     ix = int(x)
#     iy = int(y)

#     if ix == 0 and iy == 0:
#         return

#     # moveRel(xOffset, yOffset, relative=True)
#     # disable_mouse_acceleration=True
#     pydirectinput.moveRel(ix, iy, relative=True)
# ----------------------------------------------------------------

# use sendinput

# class MOUSEINPUT(ctypes.Structure):
#     _fields_ = [("dx", ctypes.c_long),
#                 ("dy", ctypes.c_long),
#                 ("mouseData", ctypes.c_ulong),
#                 ("dwFlags", ctypes.c_ulong),
#                 ("time", ctypes.c_ulong),
#                 ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

# class INPUT(ctypes.Structure):
#     _fields_ = [("type", ctypes.c_ulong),
#                 ("mi", MOUSEINPUT)]

# user32 = ctypes.windll.user32
# INPUT_MOUSE = 0
# MOUSEEVENTF_MOVE = 0x0001

# def move_mouse(x, y):
#     x = int(x)
#     y = int(y)

#     extra = ctypes.c_ulong(0)
#     ii_ = INPUT(type=INPUT_MOUSE,
#                 mi=MOUSEINPUT(dx=x,
#                               dy=y,
#                               mouseData=0,
#                               dwFlags=MOUSEEVENTF_MOVE,
#                               time=0,
#                               dwExtraInfo=ctypes.pointer(extra)))

#
#     user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))


PID_KP = 0.64
PID_KI = 0.0
PID_KD = 0.02


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # p
        P = error * self.Kp

        # I
        self.integral += error
        I = self.integral * self.Ki

        # D
        delta_error = error - self.prev_error
        D = delta_error * self.Kd

        output = P + I + D

        if output * error < 0:
            output = P

        self.prev_error = error

        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0


class EMA:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.last_val = None

    def update(self, val):
        if self.last_val is None:
            self.last_val = val
            return val

        new_val = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = new_val
        return new_val

    def reset(self):
        self.last_val = None


class Aimbot:
    def __init__(self):
        self.model = MobileNetAimBot().to(DEVICE)
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH))
        self.model.eval()
        try:
            dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)

            # 讓 Torch 跑一次模型，記錄下所有的計算路徑
            # 這會產生一個優化過的 ScriptModule
            self.model = torch.jit.trace(self.model, dummy_input)

            # 3. 強制優化 (Optional, 像是 Frozen graph)
            self.model = torch.jit.freeze(self.model)

            print("TorchScript Tracing sucessful")
        except Exception as e:
            print(f"tracing fail: {e}")
        # self.model.half()

        # try:
        #     self.model = torch.compile(self.model)
        #     print("Torch Compile activated")
        # except:
        #     pass
        self.sct = mss.mss()

        # middle of screen
        # self.screen_w = win32api.GetSystemMetrics(0)
        # self.screen_h = win32api.GetSystemMetrics(1)
        # self.center_x = self.screen_w // 2
        # self.center_y = self.screen_h // 2

        # self.monitor = {
        #     "top": self.center_y - FOV_SIZE // 2,
        #     "left": self.center_x - FOV_SIZE // 2,
        #     "width": FOV_SIZE,
        #     "height": FOV_SIZE,
        # }

        left = self.center_x - FOV_SIZE // 2
        top = self.center_y - FOV_SIZE // 2
        right = left + FOV_SIZE
        bottom = top + FOV_SIZE

        self.region = (left, top, right, bottom)

        self.camera = dxcam.create(device_idx=0, output_color="RGB")
        self.camera.start(target_fps=144, video_mode=True, region=self.region)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

        self.pid_x = PIDController(PID_KP, PID_KI, PID_KD)
        self.pid_y = PIDController(PID_KP, PID_KI, PID_KD)

        self.running = False
        print(f"截圖區域: {FOV_SIZE}*{FOV_SIZE}")

    def preprocess(self, img_np):
        # mss 截出來是 BGRA，要轉 RGB
        # img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
        img_rgb = img_np
        # 如果 FOV_SIZE 不等於 256，這裡要加 cv2.resize
        img_rgb = cv2.resize(img_rgb, (256, 256))

        # Normalize
        img_tensor = torch.from_numpy(img_rgb).to(DEVICE, non_blocking=True)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        img_tensor = img_tensor.float() / 255.0

        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    def run(self):
        # Warmup
        dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
        for _ in range(10):
            _ = self.model(dummy)
        torch.cuda.synchronize()

        EMA_ALPHA = 0.93
        DEADZONE = 2.0
        LIMIT = 50
        BASE_SENS = 0.43
        ACCEL_EXP = 1.07
        MOMENTUM_BOOST = 1.43
        OSCILLATION_DAMP = 0.6
        ema_x = EMA(EMA_ALPHA)
        ema_y = EMA(EMA_ALPHA)

        last_sign_x = 0
        last_sign_y = 0
        print("process activate, press END to terminate")
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        with torch.no_grad():
            while True:
                loop_start = time.time()
                key_state = win32api.GetKeyState(0x04)

                if key_state < 0:
                    # get input

                    # screenshot = np.array(self.sct.grab(self.monitor))
                    # screenshot=self.camera.grab(region=self.region)
                    screenshot = self.camera.get_latest_frame()
                    if screenshot is None:
                        continue

                    # model infer
                    input_tensor = self.preprocess(screenshot)

                    # temp_start=time.time()
                    coords = self.model(input_tensor)
                    # print(f"model inference time: {(time.time()-temp_start)*1000}")

                    x_norm = coords[0][0].item()
                    y_norm = coords[0][1].item()

                    raw_dx = x_norm * (FOV_SIZE / 2)
                    raw_dy = y_norm * (FOV_SIZE / 2) - 3

                    ema_dx = ema_x.update(raw_dx)
                    ema_dy = ema_y.update(raw_dy)

                    distance = (ema_dx**2 + ema_dy**2) ** 0.5

                    if distance < 1:
                        scale_factor = 0  # prevent div by 0
                    else:
                        scale_factor = BASE_SENS * (distance ** (ACCEL_EXP - 1))

                    final_dx = ema_dx * scale_factor
                    final_dy = ema_dy * scale_factor

                    curr_sign_x = 1 if final_dx > 0 else -1
                    if abs(final_dx) < DEADZONE:
                        curr_sign_x = 0

                    # same dir
                    if curr_sign_x == last_sign_x and curr_sign_x != 0:
                        final_dx *= MOMENTUM_BOOST
                    # opposite dir
                    elif curr_sign_x != last_sign_x and last_sign_x != 0:
                        final_dx *= OSCILLATION_DAMP
                    # update dir
                    last_sign_x = curr_sign_x

                    curr_sign_y = 1 if final_dy > 0 else -1
                    if abs(final_dy) < DEADZONE:
                        curr_sign_y = 0

                    if curr_sign_y == last_sign_y and curr_sign_y != 0:
                        final_dy *= MOMENTUM_BOOST
                    elif curr_sign_y != last_sign_y and last_sign_y != 0:
                        final_dy *= OSCILLATION_DAMP

                    last_sign_y = curr_sign_y

                    final_dx = min(final_dx, LIMIT)
                    final_dy = min(final_dy, LIMIT)

                    if abs(final_dx) > DEADZONE or abs(final_dy) > DEADZONE:
                        move_mouse(final_dx, final_dy)

                    latency = (time.time() - loop_start) * 1000
                    fps_counter += 1
                    # show fps
                    self.show_realtime(
                        screenshot,
                        raw_dx,
                        raw_dy,
                        final_dx,
                        final_dy,
                        current_fps,
                        latency,
                    )
                    # print(f"FPS: {int(current_fps)}   Lat: {latency:.1f}ms")

                else:
                    self.pid_x.reset()
                    self.pid_y.reset()
                    last_sign_x = 0
                    last_sign_y = 0
                    ema_x.reset()
                    ema_y.reset()

                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                if win32api.GetKeyState(0x23) < 0:
                    break
                # time.sleep(0.001)
                cv2.waitKey(1)

    # debug moniter screen
    def show_realtime(
        self, screenshot, raw_dx, raw_dy, final_dx, final_dy, fps, latency
    ):
        debug_img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR).copy()

        # center
        center_x = FOV_SIZE // 2
        center_y = FOV_SIZE // 2

        # center green
        cv2.circle(debug_img, (center_x, center_y), 3, (0, 255, 0), -1)

        # red dot, model output raw position
        try:
            pred_x = int(center_x + raw_dx)
            pred_y = int(center_y + raw_dy)

            cv2.circle(debug_img, (pred_x, pred_y), 4, (0, 0, 255), -1)
            cv2.line(debug_img, (center_x, center_y), (pred_x, pred_y), (0, 0, 255), 1)
        except Exception:
            pass

        # blue arrow show the mouse move vector
        try:
            aim_x = int(center_x + final_dx)
            aim_y = int(center_y + final_dy)

            if abs(final_dx) > 1 or abs(final_dy) > 1:
                cv2.arrowedLine(
                    debug_img,
                    (center_x, center_y),
                    (aim_x, aim_y),
                    (255, 0, 0),
                    2,
                    tipLength=0.1,
                )
        except Exception:
            pass

        dist = int((raw_dx**2 + raw_dy**2) ** 0.5)
        fps_color = (0, 255, 0)
        cv2.putText(
            debug_img,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            fps_color,
            1,
        )
        cv2.putText(
            debug_img,
            f"Lat: {latency:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            fps_color,
            1,
        )
        # cv2.putText(debug_img, f"Dist: {dist}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(
            debug_img,
            "Red: Raw target  Blue: Final vector",
            (10, FOV_SIZE - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Visualization", debug_img)


if __name__ == "__main__":
    bot = Aimbot()
    bot.run()
