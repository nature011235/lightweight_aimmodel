import matplotlib.pyplot as plt

models = ["Our Model", "YOLOv5 Nano\n(Object Det.)", "OpenPose\n(Pose Est.)"]
times = [1.58, 7.13, 52.63]
colors = ["#2ecc71", "#95a5a6", "#95a5a6"]


plt.figure(figsize=(10, 6), dpi=100)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


bars = plt.bar(models, times, color=colors, width=0.5, edgecolor="black", alpha=0.8)


plt.title("Inference Latency Comparison", fontsize=16, fontweight="bold")
plt.ylabel("Latency (ms)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)


def add_labels(bars):
    base_time = times[0]
    for i, bar in enumerate(bars):
        height = bar.get_height()
        fps = int(1000 / height)

        # (ms)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

        # compare rate
        if i > 0:
            speedup = height / base_time
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{speedup:.1f}x Slower",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=13,
            )


add_labels(bars)

plt.ylim(0, max(times) * 1.15)

plt.tight_layout()
plt.show()
