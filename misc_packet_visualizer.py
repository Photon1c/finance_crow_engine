# Generic Rubric for packet visualization
# pip install matplotlib numpy pillow
# Optional: ffmpeg on PATH for MP4 export (recommended)
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASENAME = "recursive_trade_failure_SPY"
FRAMES = 180
FPS = 15

DEFAULT_VISUAL_CONTEXT = {
    "ticker": "SPY",
    "instrument": "SPY Put",
    "spot": 0.0,
    "chain_date": "",
    "failure_type": "unknown",
    "status": "Open",
    "packet_fields": "SPY | strike | IV | delta",
    "contract_label": "SPY optimal\ncontract",
    "cloud_label": "SPY event-space",
    "raw_event_label": "SPY",
    "outcome_label": "SPY",
    "pressure_label": "legal\npressure",
}

VISUAL_CONTEXT = dict(DEFAULT_VISUAL_CONTEXT)

fig = None
ax = None

STAGE_LABELS = [
    "Reality",
    "Sanitized\nPacket",
    "Procedure\nGate",
    "Inference\nUpdate",
    "Case\nState",
    "Outcome",
]

MARGIN = 0.13
STAGE_X = [
    MARGIN + i * (1 - 2 * MARGIN) / (len(STAGE_LABELS) - 1)
    for i in range(len(STAGE_LABELS))
]
STAGES = list(zip(STAGE_LABELS, STAGE_X))

BOX_W = 0.10
BOX_H = 0.14
CHANNEL_Y = 0.48
REALITY_PACKET_Y = 0.58   # raw event packet sits above the pipeline row
PHASE_HUD_Y = 0.36        # colored phase label (REALITY, GATE, ...) below raw event
CLOUD_Y = 0.74
CLOUD_X = STAGE_X[0]

states = ["unknown", "charged", "reviewed", "routed", "resolved"]
state_scores = [0.10, 0.32, 0.48, 0.67, 0.88]

PHASE_COLORS = {
    "reality": "#566573",
    "sanitize": "#d4ac0d",
    "gate": "#c0392b",
    "inference": "#8e44ad",
    "case": "#1a5276",
    "outcome": "#117a65",
}

PINS = [
    (STAGE_X[1] + 0.035, 0.66),
    (STAGE_X[2] - 0.028, 0.60),
    (STAGE_X[2], 0.68),
    (STAGE_X[2] + 0.028, 0.59),
    (STAGE_X[2] + 0.055, 0.67),
    (STAGE_X[3] - 0.018, 0.58),
]

REALITY_CLOUD = np.random.default_rng(7).normal(
    loc=(CLOUD_X, CLOUD_Y), scale=(0.028, 0.035), size=(60, 2)
)

PHASES = [
    ("reality", 0, 30),
    ("sanitize", 30, 60),
    ("gate", 60, 100),
    ("inference", 100, 130),
    ("case", 130, 155),
    ("outcome", 155, FRAMES),
]


def configure_visual_context(context=None):
    global VISUAL_CONTEXT
    VISUAL_CONTEXT = {**DEFAULT_VISUAL_CONTEXT, **(context or {})}


def phase_at_frame(frame):
    for name, start, end in PHASES:
        if start <= frame < end:
            local = (frame - start) / max(1, end - start - 1)
            return name, local
    return PHASES[-1][0], 1.0


def phase_index(name):
    return next(i for i, (n, _, _) in enumerate(PHASES) if n == name)


def _ensure_figure():
    global fig, ax
    if fig is None or not plt.fignum_exists(fig.number):
        fig, ax = plt.subplots(figsize=(13, 7.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    return fig, ax


def _close_figure():
    global fig, ax
    if fig is not None:
        plt.close(fig)
    fig = None
    ax = None


def packet_position(phase_name, local):
    if phase_name == "reality":
        return STAGE_X[0], REALITY_PACKET_Y

    if phase_name == "sanitize":
        x = STAGE_X[0] + (STAGE_X[1] - STAGE_X[0]) * local
        y = REALITY_PACKET_Y + (CHANNEL_Y - REALITY_PACKET_Y) * local
        return x, y

    if phase_name == "gate":
        x = STAGE_X[1] + (STAGE_X[3] - STAGE_X[1]) * local
        y = CHANNEL_Y + 0.10 * np.sin(local * np.pi * 5)
        return x, y

    if phase_name == "inference":
        x = STAGE_X[3] + (STAGE_X[4] - STAGE_X[3]) * local
        return x, CHANNEL_Y

    if phase_name == "case":
        x = STAGE_X[4] + (STAGE_X[5] - STAGE_X[4]) * local
        return x, CHANNEL_Y

    return STAGE_X[5], CHANNEL_Y


def pipeline_progress(phase_name, local):
    idx = phase_index(phase_name)
    return (idx + local) / (len(PHASES) - 1)


def draw_pipeline(active_idx):
    ctx = VISUAL_CONTEXT
    ticker = ctx.get("ticker", "SPY")
    contract = ctx.get("optimal_contract") or {}
    title = (
        f"{ticker} Recursive Trade Failure @ ${ctx.get('spot', 0):.2f} — "
        f"{ctx.get('instrument', ticker)}"
    )
    ax.text(
        0.5, 0.955,
        title,
        ha="center", va="center", fontsize=14, weight="bold", zorder=10,
    )
    if ctx.get("chain_date"):
        ax.text(
            0.5, 0.925,
            f"chain {ctx['chain_date']}  |  optimal {contract.get('symbol', '')}",
            ha="center", va="center", fontsize=9, color="#566573", zorder=10,
        )

    for idx, (label, x) in enumerate(STAGES):
        active = idx == active_idx
        rect = Rectangle(
            (x - BOX_W / 2, CHANNEL_Y - BOX_H / 2),
            BOX_W, BOX_H,
            fill=active,
            facecolor="#d6eaf8" if active else "white",
            edgecolor="#1a5276" if active else "black",
            lw=3.5 if active else 2.0,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.text(
            x, CHANNEL_Y, label,
            ha="center", va="center", fontsize=10, linespacing=1.2,
            weight="bold" if active else "normal", zorder=5,
        )

    for i in range(len(STAGES) - 1):
        x1 = STAGES[i][1] + BOX_W / 2 + 0.01
        x2 = STAGES[i + 1][1] - BOX_W / 2 - 0.01
        ax.add_patch(
            FancyArrowPatch(
                (x1, CHANNEL_Y), (x2, CHANNEL_Y),
                arrowstyle="->", mutation_scale=14, lw=1.8, color="#333333", zorder=3,
            )
        )

    ax.text(
        STAGE_X[2], 0.78,
        "rubrics · schedules · guidelines · deadlines · eligibility",
        ha="center", fontsize=10, zorder=5,
    )

    ax.text(0.5, 0.22, "Inferred State Updates", ha="center", fontsize=12, weight="bold", zorder=5)
    ax.plot([0.16, 0.80], [0.16, 0.16], lw=2, color="black", zorder=3)
    state_labels = states
    if ctx.get("failure_type") and ctx["failure_type"] != "unknown":
        state_labels = states[:-1] + [ctx["failure_type"][:10]]
    for s, score in zip(state_labels, state_scores):
        sx = 0.16 + score * 0.64
        ax.plot([sx, sx], [0.14, 0.18], lw=2, color="black", zorder=3)
        ax.text(sx, 0.10, s, ha="center", fontsize=9, zorder=5)


def draw_phase_hud(phase_name, local, frame):
    ctx = VISUAL_CONTEXT
    color = PHASE_COLORS[phase_name]
    pulse = 0.004 * np.sin(2 * np.pi * frame / 12)
    hud_w, hud_h = 0.26, 0.055
    hud_left = STAGE_X[0] - hud_w / 2
    hud_text = phase_name.upper()
    if ctx.get("ticker"):
        hud_text = f"{ctx['ticker']} · {hud_text}"

    ax.add_patch(
        FancyBboxPatch(
            (hud_left, PHASE_HUD_Y - hud_h / 2), hud_w, hud_h,
            boxstyle="round,pad=0.012",
            facecolor=color, edgecolor="black", lw=1.5, alpha=0.92, zorder=12,
        )
    )
    ax.text(
        STAGE_X[0], PHASE_HUD_Y, hud_text,
        ha="center", va="center", fontsize=10, weight="bold", color="white", zorder=13,
    )

    bar_left, bar_width = 0.28, 0.58
    prog = pipeline_progress(phase_name, local)
    ax.add_patch(Rectangle((bar_left, 0.895), bar_width, 0.022, fill=False, lw=1.5, zorder=11))
    ax.add_patch(
        Rectangle(
            (bar_left, 0.895), bar_width * prog + pulse, 0.022,
            facecolor=color, edgecolor=color, alpha=0.85, zorder=11,
        )
    )
    ax.text(bar_left + bar_width / 2, 0.872, "pipeline progress", ha="center", fontsize=8, zorder=11)


def draw_pins(highlight, packet_x, packet_y):
    for px, py in PINS:
        near = highlight and abs(packet_x - px) < 0.04 and abs(packet_y - py) < 0.06
        color = "#c0392b" if near else ("#e74c3c" if highlight else "#666666")
        lw = 3.2 if near else (2.4 if highlight else 1.5)
        radius = 0.018 if near else 0.013
        ax.add_patch(Circle((px, py), radius, fill=near, facecolor="#fadbd8", edgecolor=color, lw=lw, zorder=6))
        if near:
            ax.plot([px, packet_x], [py, packet_y], color="#c0392b", lw=1.2, ls="--", zorder=5)


def inferred_state_x(phase_name, local):
    if phase_name in ("reality", "sanitize"):
        score = state_scores[0]
    elif phase_name == "gate":
        score = state_scores[1]
    elif phase_name == "inference":
        score = state_scores[1] + local * (state_scores[3] - state_scores[1])
    elif phase_name == "case":
        score = state_scores[3] + local * (state_scores[4] - state_scores[3])
    else:
        score = state_scores[4]
    return 0.16 + score * 0.64


def pressure_level(phase_name, local):
    if phase_name in ("reality", "sanitize"):
        return 0.18
    if phase_name == "gate":
        return 0.50 + 0.40 * abs(np.sin(np.pi * local))
    if phase_name == "inference":
        return 0.88
    if phase_name == "case":
        return 0.62
    return 0.22


def update(frame):
    _, ax = _ensure_figure()
    ctx = VISUAL_CONTEXT
    ax.clear()
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    phase_name, local = phase_at_frame(frame)
    active_idx = min(phase_index(phase_name), len(STAGES) - 1)
    draw_pipeline(active_idx)
    draw_phase_hud(phase_name, local, frame)

    x, y = packet_position(phase_name, local)

    if phase_name in ("reality", "sanitize"):
        fade = 1.0 if phase_name == "reality" else max(0.0, 1.0 - local)
        if fade > 0.05:
            keep = max(1, int(len(REALITY_CLOUD) * fade))
            drift = 0.015 * np.sin(2 * np.pi * frame / 12)
            cloud_pts = REALITY_CLOUD[:keep].copy()
            if phase_name == "sanitize":
                shrink = 1.0 - 0.65 * local
                cloud_pts[:, 0] = CLOUD_X + (cloud_pts[:, 0] - CLOUD_X) * shrink
                cloud_pts[:, 1] = CLOUD_Y + (cloud_pts[:, 1] - CLOUD_Y) * shrink
            cloud_pts[:, 0] += drift
            cloud_pts[:, 1] += 0.010 * np.cos(2 * np.pi * frame / 16)
            ax.scatter(cloud_pts[:, 0], cloud_pts[:, 1], s=22, c="#7f8c8d", zorder=1)
            ax.text(
                CLOUD_X, 0.86, ctx.get("cloud_label", "messy event-space"),
                ha="center", fontsize=9, color="#566573", zorder=2,
            )
            ax.plot(
                [0.04, STAGE_X[0] + BOX_W / 2 + 0.04], [0.64, 0.64],
                ls=":", lw=1.2, color="#95a5a6", zorder=2,
            )

    if phase_name == "sanitize":
        divider_x = STAGE_X[0] + (STAGE_X[1] - STAGE_X[0]) * 0.5
        glow = 0.5 + 0.5 * np.sin(2 * np.pi * frame / 8)
        ax.add_patch(
            Rectangle(
                (divider_x - 0.010, CHANNEL_Y - BOX_H / 2 - 0.02),
                0.020, BOX_H + 0.04,
                facecolor="#f9e79f", edgecolor="#d4ac0d", lw=2.5, alpha=0.35 + 0.45 * glow, zorder=6,
            )
        )
        ax.text(divider_x, CHANNEL_Y + BOX_H / 2 + 0.05, "sanitize", ha="center", fontsize=9, color="#7d6608", weight="bold", zorder=7)
        for i in range(6):
            src_y = CLOUD_Y - 0.04 + i * 0.015
            prog = min(1.0, local * 1.1 + i * 0.06)
            mid_x = CLOUD_X + (x - CLOUD_X) * prog
            mid_y = src_y + (REALITY_PACKET_Y - src_y) * prog
            if local > 0.35:
                mid_y += (CHANNEL_Y - REALITY_PACKET_Y) * (local - 0.35) / 0.65
            ax.annotate(
                "", xy=(mid_x, mid_y), xytext=(CLOUD_X, src_y),
                arrowprops=dict(arrowstyle="->", color="#d4ac0d", lw=1.4),
                zorder=6,
            )

    if phase_name == "gate":
        path_x = np.linspace(STAGE_X[1], STAGE_X[3], 40)
        path_y = CHANNEL_Y + 0.10 * np.sin(np.linspace(0, np.pi, 40) * 5)
        ax.plot(path_x, path_y, color="#fadbd8", lw=3.0, alpha=0.8, zorder=2)
        ax.plot(path_x, path_y, color="#c0392b", lw=1.2, ls="--", zorder=3)

    draw_pins(highlight=(phase_name == "gate"), packet_x=x, packet_y=y)

    sanitized = phase_name not in ("reality",)
    packet_w = 0.072 if sanitized else 0.052
    packet_h = 0.050 if sanitized else 0.042
    packet_color = "#117a65" if sanitized else "#566573"
    ax.add_patch(
        Rectangle(
            (x - packet_w / 2, y - packet_h / 2), packet_w, packet_h,
            fill=True, facecolor="#d5f5e3" if sanitized else "#ecf0f1",
            edgecolor=packet_color, lw=3.0, zorder=8,
        )
    )
    ax.text(
        x, y, ctx.get("raw_event_label", "raw event") if not sanitized else "packet",
        ha="center", va="center", fontsize=8 if sanitized else 9,
        weight="bold", color=packet_color, zorder=9,
    )
    if sanitized:
        ax.text(
            x, y - 0.075, ctx.get("packet_fields", "charge | date | status | score"),
            ha="center", fontsize=7, color="#117a65", zorder=9,
        )

    inferred = inferred_state_x(phase_name, local)
    if phase_name in ("inference", "case", "outcome"):
        ax.annotate(
            "", xy=(inferred, 0.16), xytext=(x, y - 0.05),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=2.0, connectionstyle="arc3,rad=-0.2"),
            zorder=6,
        )
    ax.add_patch(
        Circle((inferred, 0.16), 0.020, fill=True, facecolor="#ebdef0", edgecolor="#8e44ad", lw=3.0, zorder=7)
    )
    ax.text(inferred, 0.205, "current estimate", ha="center", fontsize=8, color="#8e44ad", weight="bold", zorder=8)

    pressure = pressure_level(phase_name, local)
    gauge_x, gauge_bottom, gauge_h = 0.965, 0.30, 0.52
    ax.add_patch(Rectangle((gauge_x, gauge_bottom), 0.022, gauge_h, fill=False, lw=1.8, zorder=4))
    ax.add_patch(
        Rectangle(
            (gauge_x, gauge_bottom), 0.022, gauge_h * pressure,
            facecolor="#e74c3c", edgecolor="#c0392b", lw=1.0, zorder=5,
        )
    )
    ax.text(
        gauge_x + 0.011, gauge_bottom + gauge_h + 0.03,
        ctx.get("pressure_label", "legal\npressure"),
        ha="center", fontsize=8, zorder=5,
    )

    if phase_name == "outcome":
        pulse = 0.015 + 0.015 * np.sin(2 * np.pi * frame / 10)
        ax.add_patch(
            Rectangle(
                (STAGE_X[5] - BOX_W / 2 - pulse, CHANNEL_Y - BOX_H / 2 - pulse),
                BOX_W + 2 * pulse, BOX_H + 2 * pulse,
                fill=False, edgecolor="#117a65", lw=3.0, zorder=7,
            )
        )
        ax.text(
            STAGE_X[5], CHANNEL_Y - BOX_H / 2 - 0.05,
            ctx.get("contract_label", ctx.get("outcome_label", "outcome")),
            ha="center", va="top", fontsize=7, color="#117a65", weight="bold", zorder=8,
        )

    captions = {
        "reality": f"{ctx.get('ticker', 'SPY')} full tape — spot ${ctx.get('spot', 0):.2f}",
        "sanitize": f"Compress to optimal {ctx.get('outcome_label', 'contract')}",
        "gate": "Rubrics and deadlines deflect the route",
        "inference": f"Inferred failure: {ctx.get('failure_type', 'unknown')}",
        "case": f"Status: {ctx.get('status', 'Open')}",
        "outcome": f"Resolved via {ctx.get('outcome_label', 'contract')}",
    }
    ax.text(0.5, 0.035, captions[phase_name], ha="center", fontsize=10, style="italic", zorder=10)
    # Keep every PNG frame unique so GIF encoders retain full animation timing.
    ax.text(
        0.005, 0.005, f"{frame:04d}", fontsize=1, color="#fefefe",
        alpha=0.01, zorder=0,
    )


def render_frame(frame, png_path):
    update(frame)
    fig, _ = _ensure_figure()
    fig.canvas.draw()
    fig.savefig(png_path, dpi=120, facecolor="white", edgecolor="white")


def render_all_frames(tmpdir, frames):
    paths = []
    for frame in range(frames):
        path = os.path.join(tmpdir, f"frame_{frame:04d}.png")
        render_frame(frame, path)
        paths.append(path)
    return paths


def save_mp4(frame_dir, frames, fps, path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", "0", "-i", pattern,
        "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"MP4 export failed:\n{exc.stderr.strip()}", file=sys.stderr)
        if os.path.isfile(path):
            os.remove(path)
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 1024


def save_gif_ffmpeg(frame_dir, frames, fps, path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", "0", "-i", pattern,
        "-frames:v", str(frames),
        "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer",
        "-loop", "0", path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"GIF export via ffmpeg failed:\n{exc.stderr.strip()}", file=sys.stderr)
        if os.path.isfile(path):
            os.remove(path)
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 1024


def save_gif(frame_paths, fps, path):
    """Write GIF with all frames and animation timing preserved."""
    frame_dir = os.path.dirname(frame_paths[0])
    frames = len(frame_paths)

    if save_gif_ffmpeg(frame_dir, frames, fps, path):
        return

    from PIL import Image

    images = []
    for i, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as img:
            rgb = img.convert("RGB")
            px = rgb.load()
            r, g, b = px[0, 0]
            px[0, 0] = ((r + i) % 250, g, b)
            images.append(rgb.copy())

    duration_ms = max(1, int(1000 / fps))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def verify_gif(path, expected_frames):
    from PIL import Image

    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(f"Output missing or empty: {path}")

    with Image.open(path) as im:
        count = getattr(im, "n_frames", 1)
        im.seek(0)
        first = im.copy()
        im.seek(min(count - 1, expected_frames - 1))
        last = im.copy()

    if count != expected_frames:
        raise RuntimeError(f"{path}: expected {expected_frames} frames, got {count}")

    if first.tobytes() == last.tobytes():
        raise RuntimeError(f"{path}: first and last frames are identical — animation did not render.")


def verify_mp4(path, expected_frames, fps):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(f"Output missing or empty: {path}")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return

    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    count = int(result.stdout.strip())
    if count != expected_frames:
        raise RuntimeError(f"{path}: expected {expected_frames} frames, got {count}")


def render_packet_animation(output_dir=None, frames=FRAMES, fps=FPS, basename=DEFAULT_BASENAME, context=None):
    """Render constraint-architecture animation; returns list of output file paths."""
    configure_visual_context(context)
    out_dir = output_dir or SCRIPT_DIR
    out_mp4 = os.path.join(out_dir, f"{basename}.mp4")
    out_gif = os.path.join(out_dir, f"{basename}.gif")

    _ensure_figure()
    tmpdir = tempfile.mkdtemp(prefix="packet_viz_")
    outputs = []
    try:
        print(f"Rendering {frames} frames...", flush=True)
        frame_paths = render_all_frames(tmpdir, frames)

        if save_mp4(tmpdir, frames, fps, out_mp4):
            verify_mp4(out_mp4, frames, fps)
            outputs.append(out_mp4)

        save_gif(frame_paths, fps, out_gif)
        verify_gif(out_gif, frames)
        outputs.append(out_gif)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _close_figure()

    for path in outputs:
        print(f"saved: {path} ({frames} frames @ {fps} fps)")

    if out_mp4 in outputs:
        print("Tip: open the .mp4 for the most reliable playback of all animation effects.")
    if out_gif in outputs:
        print(f"GIF ready: {out_gif}")

    return outputs


def main():
    render_packet_animation()


if __name__ == "__main__":
    main()
