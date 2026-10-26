"""Image grids, trajectories, and MMD plots shared by all experiments."""
from pathlib import Path
import math
import os
import numpy as np
from PIL import Image, ImageDraw
import torch


def save_grid(images: torch.Tensor, path: Path, count: int = 64,
              image_shape: tuple[int, int, int] = (1, 28, 28)) -> None:
    # Convert generator outputs to display pixels.
    channels, height, width = image_shape
    pixels = images[:count].detach().cpu().reshape(-1, *image_shape).clamp(0, 1)
    pixels = (pixels * 255).round().to(torch.uint8).numpy()
    pixels = pixels[:, 0] if channels == 1 else pixels.transpose(0, 2, 3, 1)
    columns = min(8, len(pixels))
    rows = math.ceil(len(pixels) / columns)
    canvas_shape = (rows * (height + 2), columns * (width + 2))
    canvas = np.zeros(canvas_shape if channels == 1 else (*canvas_shape, channels), dtype=np.uint8)
    for index, digit in enumerate(pixels):
        row, column = divmod(index, columns)
        top, left = row * (height + 2), column * (width + 2)
        canvas[top:top + height, left:left + width] = digit
    Image.fromarray(canvas).resize((columns * 120, rows * 120), Image.Resampling.NEAREST).save(path)



def save_trajectory(states: list[torch.Tensor], iterations: list[int], output: Path,
                    count: int, unit: str) -> None:
    particles = torch.stack(states)
    count = min(count, particles.shape[1])
    channels = particles.shape[2]
    canvas = Image.new("L" if channels == 1 else "RGB", (80 + 120 * len(iterations), 25 + 120 * count))
    text_color = 255 if channels == 1 else (255, 255, 255)
    draw = ImageDraw.Draw(canvas)
    for column, iteration in enumerate(iterations):
        draw.text((80 + column * 120, 5), f"{unit} {iteration}", fill=text_color)
        for row in range(count):
            sample = particles[column, row].clamp(0, 1)
            sample = sample[0] if channels == 1 else sample.permute(1, 2, 0)
            pixels = (sample * 255).round().to(torch.uint8).numpy()
            digit = Image.fromarray(pixels).resize((112, 112), Image.Resampling.NEAREST)
            canvas.paste(digit, (80 + column * 120, 25 + row * 120))
    for row in range(count):
        draw.text((5, 25 + row * 120 + 50), f"particle {row}", fill=text_color)
    canvas.save(output / "trajectory.png")



def plot_mmd(history: list[tuple[int, float, float]], output: Path, unit: str,
             *, fid_history: list[tuple[int, float]] | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist_riesz_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "serif", "font.size": 14,
                         "axes.labelsize": 18, "lines.linewidth": 2,
                         "legend.frameon": False, "xtick.direction": "in",
                         "ytick.direction": "in"}):
        has_fid = bool(fid_history)
        fig, axes = plt.subplots(1, 2 if has_fid else 1,
                                 figsize=(13, 4.5) if has_fid else (6, 4), squeeze=False)
        ax = axes[0, 0]
        values = np.asarray(history)
        train = np.sqrt(np.maximum(values[:, 1], 0))
        heldout = np.sqrt(np.maximum(values[:, 2], 0))
        ax.plot(values[:, 0], train, color="steelblue", label="Training subset")
        ax.plot(values[:, 0], heldout, color="steelblue", linestyle="--", label="Heldout test subset")
        ax.set_xlabel("Iteration (full training epoch)" if unit == "epoch" else "Iteration (update)")
        ax.set_ylabel(r"$\mathrm{MMD}$")
        if np.all(train > 0) and np.all(heldout > 0):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        if has_fid:
            fid_ax = axes[0, 1]
            fid_values = np.asarray(fid_history)
            fid_ax.plot(fid_values[:, 0], fid_values[:, 1], color="darkorange")
            fid_ax.set_xlabel("Iteration (full training epoch)" if unit == "epoch" else "Iteration (update)")
            fid_ax.set_ylabel("FID")
            fid_ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = "fid_vs_iteration.png" if has_fid else "mmd_vs_iterations.png"
        fig.savefig(output / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

