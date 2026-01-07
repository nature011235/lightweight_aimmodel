import matplotlib.pyplot as plt
import numpy as np

EMA_ALPHA = 0.9
BASE_SENS = 0.41
ACCEL_EXP = 1.07
MOMENTUM_BOOST = 1.3
OSCILLATION_DAMP = 0.6
DEADZONE = 0.0


TOTAL_FRAMES = 60
TARGET_POS = 100.0
NOISE_LEVEL = 4.0


class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_val = None

    def update(self, val):
        if self.last_val is None:
            self.last_val = val
            return val
        new_val = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = new_val
        return new_val


def simulate():
    history_target = []
    history_pos_low = []
    history_pos_high = []
    history_pos_mid = []
    history_pos_momentum = []

    pos_low = 0.0
    pos_high = 0.0
    pos_mid = 0.0
    pos_momentum = 0.0

    ema = EMA(EMA_ALPHA)
    last_sign = 0

    for i in range(TOTAL_FRAMES):
        target = TARGET_POS if i >= 5 else 0.0
        if i >= 30:
            target = -TARGET_POS

        noise = np.random.normal(0, NOISE_LEVEL) if i >= 5 else 0
        perceived_target = target + noise
        history_target.append(target)

        # --- Group A: Low Sens (0.3) ---
        err = perceived_target - pos_low
        pos_low += err * 0.4
        history_pos_low.append(pos_low)

        # --- Group B: High Sens (1.1) ---
        err = perceived_target - pos_high
        pos_high += err * 1.1
        history_pos_high.append(pos_high)

        # --- Group C: Mid Sens (0.7)
        err = perceived_target - pos_mid
        pos_mid += err * 0.7
        history_pos_mid.append(pos_mid)

        # --- Group D: our way ---
        raw_dx = perceived_target - pos_momentum
        ema_dx = ema.update(raw_dx)
        distance = abs(ema_dx)
        if distance < 1:
            scale = 0
        else:
            scale = BASE_SENS * (distance ** (ACCEL_EXP - 1))

        final_dx = ema_dx * scale
        curr_sign = 1 if final_dx > 0 else -1
        if abs(final_dx) < 0.1:
            curr_sign = 0

        if curr_sign == last_sign and curr_sign != 0:
            final_dx *= MOMENTUM_BOOST
        elif curr_sign != last_sign and last_sign != 0:
            final_dx *= OSCILLATION_DAMP
        last_sign = curr_sign

        pos_momentum += final_dx
        history_pos_momentum.append(pos_momentum)

    return (
        history_target,
        history_pos_low,
        history_pos_high,
        history_pos_mid,
        history_pos_momentum,
    )


target, low, high, mid, momentum = simulate()
frames = range(TOTAL_FRAMES)

plt.figure(figsize=(10, 6), dpi=100)


plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


plt.plot(frames, target, "k--", label="Enemy Target", alpha=0.3, linewidth=2)

plt.plot(frames, low, "r-", label="Linear Control(low)", alpha=0.5, linewidth=1.5)

# plt.plot(frames, mid, color='orange', linestyle='-', label='Linear Control(mid)', alpha=0.6, linewidth=1.5)

plt.plot(frames, high, "g-", label="Linear Control(high)", alpha=0.5, linewidth=1.5)

plt.plot(frames, momentum, "b-", label="Ours", linewidth=1.5)


plt.title("Control Logic Comparison: Step Response", fontsize=16, fontweight="bold")
plt.xlabel("Step", fontsize=12)
plt.ylabel("X axis Distance", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper right", fontsize=10)


# plt.annotate('Noise Suppression (抗噪)', xy=(50, 200), xytext=(35, 230),
#              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='blue')

# plt.annotate('Fast Rise Time (快速響應)', xy=(10, 100), xytext=(15, 50),
#              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='blue')

plt.tight_layout()
plt.show()
