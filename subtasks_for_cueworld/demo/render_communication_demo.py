"""CommunicationEnv の依存ありと yoke を、同じ幼児の行動列で録画する。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from envs.communication import CommunicationEnv

FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
FONT = font_manager.FontProperties(fname=FONT_PATH)

HERE = Path(__file__).resolve().parent
GIF_PATH = HERE / "communication_demo.gif"
PNG_PATH = HERE / "communication_timeline.png"

# 行動は記号 0–3（noop なし）。有効な 0 のあとに決まった親 2 が来ることを見せる。
INFANT_ACTIONS = [0, 0, 1, 1, 2, 3, 0, 0, 1, 2, 0, 3]

COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#B279A2",
}
SYMBOL_NAME = {0: "0", 1: "1", 2: "2", 3: "3"}


def rollout(env: CommunicationEnv, actions: list[int], *, options=None):
    env.reset(seed=0, options=options)
    rows = []
    for action in actions:
        obs, _, _, _, info = env.step(action)
        rows.append(
            {
                "infant": action,
                "parent": info["parent_symbol"],
                "phase": info["phase"],
                "yoke": info["yoke"],
                "obs_argmax": int(np.argmax(obs)),
            }
        )
    return rows


def record_parent_symbols(n: int) -> np.ndarray:
    """別エピソードで親記号列を記録する（幼児はずっと記号 1）。"""
    env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=1)
    env.reset(seed=1)
    tape = []
    for _ in range(n):
        _, _, _, _, info = env.step(1)
        tape.append(info["parent_symbol"])
    return np.asarray(tape, dtype=np.int64)


def caption(row: dict) -> str:
    if row["yoke"]:
        return "yoke。親の記号列は別エピソードの再生なので、いまの行動では次が決まらない。"
    if row["infant"] == 0:
        return "有効な記号 0。親は決まった返事 2 を返す。"
    return f"無効な記号 {row['infant']}。親は 4 記号から一様に出す。"


def draw_frame(records: list[dict], t: int, phase: str, caption_text: str):
    n = len(records)
    fig_w = max(11.5, 1.05 * n + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=120)
    ax.set_xlim(-1.7, n + 0.3)
    ax.set_ylim(-1.0, 4.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        n / 2 - 0.5,
        3.7,
        phase,
        ha="center",
        va="center",
        fontsize=16,
        fontproperties=FONT,
        color="#222222",
    )
    ax.text(
        n / 2 - 0.5,
        3.15,
        caption_text,
        ha="center",
        va="center",
        fontsize=11,
        fontproperties=FONT,
        color="#333333",
        wrap=True,
    )

    row_y = {"infant": 1.9, "parent": 0.7}
    ax.text(-1.55, row_y["infant"] + 0.28, "幼児", fontproperties=FONT, fontsize=13, va="center")
    ax.text(-1.55, row_y["parent"] + 0.28, "親", fontproperties=FONT, fontsize=13, va="center")

    current = records[t]
    for i, row in enumerate(records):
        filled = i <= t
        for key, action_key in (("infant", "infant"), ("parent", "parent")):
            action = row[action_key] if filled else None
            color = COLORS[action] if action is not None else "white"
            edge = "#222222" if i == t else "#BBBBBB"
            width = 2.2 if i == t else 0.8
            y = row_y[key]
            patch = FancyBboxPatch(
                (i, y),
                0.86,
                0.72,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor=edge,
                linewidth=width,
            )
            ax.add_patch(patch)
            if filled:
                ax.text(
                    i + 0.43,
                    y + 0.36,
                    SYMBOL_NAME[action],
                    ha="center",
                    va="center",
                    color="#FFFFFF",
                    fontsize=13,
                    fontproperties=FONT,
                )
        ax.text(
            i + 0.43,
            0.28,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="#888888",
            fontproperties=FONT,
        )

    ax.text(
        -1.55,
        -0.35,
        f"親の記号: {current['parent']}    観測 argmax: {current['obs_argmax']}",
        fontproperties=FONT,
        fontsize=12,
        color="#222222",
    )

    legend_x = n - 4.3
    for j, (name, color) in enumerate(
        (
            ("0", COLORS[0]),
            ("1", COLORS[1]),
            ("2", COLORS[2]),
            ("3", COLORS[3]),
        )
    ):
        x = legend_x + j * 1.05
        ax.add_patch(
            Rectangle((x, -0.8), 0.32, 0.32, facecolor=color, edgecolor="#888888")
        )
        ax.text(x + 0.4, -0.64, name, fontproperties=FONT, fontsize=10, va="center")

    fig.tight_layout()
    return fig


def save_timeline(dependent: list[dict], yoke: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.2), dpi=140)
    fig.patch.set_facecolor("white")
    for ax, rows, title in (
        (axes[0], dependent, "依存あり。有効な記号 0 の次は親 2、それ以外は一様"),
        (axes[1], yoke, "yoke。同じ幼児の行動でも、親の記号列は別エピソードの再生"),
    ):
        ax.set_xlim(-1.5, len(rows) + 0.2)
        ax.set_ylim(-0.35, 2.15)
        ax.axis("off")
        ax.set_title(title, fontproperties=FONT, fontsize=12, loc="left", color="#222222")
        for i, row in enumerate(rows):
            for key, y in (("infant", 1.15), ("parent", 0.15)):
                action = row[key]
                ax.add_patch(
                    FancyBboxPatch(
                        (i, y),
                        0.86,
                        0.72,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=COLORS[action],
                        edgecolor="#CCCCCC",
                        linewidth=0.6,
                    )
                )
                ax.text(
                    i + 0.43,
                    y + 0.36,
                    SYMBOL_NAME[action],
                    ha="center",
                    va="center",
                    color="#FFFFFF",
                    fontsize=11,
                    fontproperties=FONT,
                )
        ax.text(-1.35, 1.5, "幼児", fontproperties=FONT, fontsize=11, va="center")
        ax.text(-1.35, 0.5, "親", fontproperties=FONT, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def main() -> None:
    n = len(INFANT_ACTIONS)
    env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=0)
    dependent = rollout(env, INFANT_ACTIONS)

    tape = record_parent_symbols(n)
    yoke_env = CommunicationEnv(max_steps=n, n_meaningful=1, seed=0)
    yoke = rollout(yoke_env, INFANT_ACTIONS, options={"yoke_parent_symbols": tape})

    save_timeline(dependent, yoke)

    frames = []
    sequences = (
        (dependent, "依存あり"),
        (yoke, "yoke（親の記号は別エピソードの再生）"),
    )
    for records, phase in sequences:
        for t, row in enumerate(records):
            text = caption(row)
            fig = draw_frame(records, t, phase, text)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf.copy())
            plt.close(fig)

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=1100,
        loop=0,
        disposal=2,
    )
    print(f"wrote {GIF_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
