from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "DroneFace" / "open_data_set"
OUT_DIR = ROOT / "Informe" / "figs" / "dataset"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _cover_square(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(img)


def _text_center(draw: ImageDraw.ImageDraw, box, text: str, font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def _draw_drone(ax, x: float, y: float, color: str) -> None:
    arm = "#64748b"
    body = "#bfdbfe"
    dark = "#1e3a8a"

    ax.plot([x - 0.72, x + 0.72], [y + 0.05, y + 0.05], color=arm, linewidth=2.0, zorder=5)
    ax.plot([x, x], [y - 0.08, y - 0.34], color=arm, linewidth=2.0, zorder=5)

    ax.add_patch(
        FancyBboxPatch(
            (x - 0.32, y - 0.10),
            0.64,
            0.26,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=body,
            edgecolor=color,
            linewidth=1.8,
            zorder=6,
        )
    )
    ax.add_patch(Rectangle((x - 0.14, y - 0.34), 0.28, 0.18, facecolor="#e0f2fe", edgecolor=arm, linewidth=1.2, zorder=6))
    ax.add_patch(Circle((x, y - 0.25), 0.07, facecolor=dark, edgecolor="#eff6ff", linewidth=1.0, zorder=7))

    for dx in (-0.72, 0.72):
        ax.add_patch(Circle((x + dx, y + 0.05), 0.10, facecolor="#dbeafe", edgecolor=arm, linewidth=1.2, zorder=7))
        ax.add_patch(Ellipse((x + dx - 0.13, y + 0.08), 0.34, 0.07, angle=8, facecolor=color, edgecolor=color, alpha=0.9, zorder=6))
        ax.add_patch(Ellipse((x + dx + 0.13, y + 0.08), 0.34, 0.07, angle=-8, facecolor=color, edgecolor=color, alpha=0.9, zorder=6))


def make_pose_collage(source: str, subjects=("a", "b")) -> Path:
    poses = [("porL", "Lateral porL"), ("porF", "Frontal porF"), ("porR", "Lateral porR")]
    cell = 300
    row_label_w = 130
    header_h = 78
    title_h = 70
    gap = 20
    margin = 34
    width = margin * 2 + row_label_w + len(poses) * cell + (len(poses) - 1) * gap
    height = margin * 2 + title_h + header_h + len(subjects) * cell + (len(subjects) - 1) * gap

    bg = (248, 250, 252)
    ink = (31, 41, 55)
    muted = (75, 85, 99)
    accent = (37, 99, 235)
    line = (209, 213, 219)
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    title_font = _font(34, bold=True)
    label_font = _font(24, bold=True)
    small_font = _font(18, bold=False)
    _text_center(
        draw,
        (0, margin, width, margin + title_h),
        f"DroneFace {source}: referencias por pose",
        title_font,
        ink,
    )

    x0 = margin + row_label_w
    y_header = margin + title_h
    for col, (_, label) in enumerate(poses):
        x = x0 + col * (cell + gap)
        draw.rounded_rectangle((x, y_header + 8, x + cell, y_header + header_h - 8), radius=8, fill=(239, 246, 255), outline=(191, 219, 254))
        _text_center(draw, (x, y_header + 8, x + cell, y_header + header_h - 8), label, label_font, accent)

    y0 = y_header + header_h
    for row, subject in enumerate(subjects):
        y = y0 + row * (cell + gap)
        draw.rounded_rectangle((margin, y, margin + row_label_w - 18, y + cell), radius=8, fill=(255, 255, 255), outline=line)
        _text_center(draw, (margin, y, margin + row_label_w - 18, y + cell), f"Sujeto\n{subject}", label_font, muted)
        for col, (pose, _) in enumerate(poses):
            suffix = "jpg" if pose == "porF" else ("jpg" if source == "trio_cam" else "JPG")
            prefix = "cam" if source == "trio_cam" else "gp"
            path = DATASET / source / f"{subject}_{prefix}_0_{pose}_00.{suffix}"
            if not path.exists() and suffix == "JPG":
                path = path.with_suffix(".jpg")
            img = _cover_square(path, cell)
            x = x0 + col * (cell + gap)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(17, 24, 39), width=2)
            draw.rectangle((x, y + cell - 34, x + cell, y + cell), fill=(17, 24, 39))
            _text_center(draw, (x, y + cell - 34, x + cell, y + cell), path.name, small_font, (255, 255, 255))

    out = OUT_DIR / f"droneface_{source}_pose_collage.png"
    canvas.save(out, quality=95)
    return out


def make_capture_geometry() -> Path:
    out = OUT_DIR / "droneface_capture_geometry_schematic.png"
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=180)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    ax.plot([0, 17.6], [0, 0], color="#111827", linewidth=2.5)
    for x in np.arange(0, 17.5, 1.0):
        tick_h = 0.18 if x % 2 else 0.28
        ax.plot([x, x], [0, -tick_h], color="#111827", linewidth=1.7)
    for x, label in [(0, "0 m"), (2, "2 m"), (7, "7 m"), (12, "12 m"), (17, "17 m")]:
        ax.text(x, -0.55, label, ha="center", va="top", fontsize=12, color="#111827")

    # Person icon at the origin.
    ax.scatter([0], [1.55], s=180, color="#7f1d1d", zorder=4)
    ax.plot([0, 0], [0.45, 1.35], color="#7f1d1d", linewidth=5, solid_capstyle="round")
    ax.plot([-0.25, 0.25], [1.05, 1.05], color="#7f1d1d", linewidth=4, solid_capstyle="round")
    ax.plot([0, -0.22], [0.45, 0.0], color="#7f1d1d", linewidth=4, solid_capstyle="round")
    ax.plot([0, 0.22], [0.45, 0.0], color="#7f1d1d", linewidth=4, solid_capstyle="round")
    ax.text(0.28, 1.45, "sujeto", fontsize=12, color="#7f1d1d", va="center")

    heights = [1.5, 3, 4, 5]
    distances = [2.5, 7.0, 12.0, 16.5]
    colors = ["#b91c1c", "#dc2626", "#f59e0b", "#2563eb"]

    for h, d, c in zip(heights, distances, colors):
        ax.plot([d, d], [0, h], color="#9ca3af", linestyle=":", linewidth=1.4)
        _draw_drone(ax, d, h, c)
        ax.annotate(
            "",
            xy=(0.28, 1.55),
            xytext=(d - 0.28, h),
            arrowprops=dict(arrowstyle="-", color="#9ca3af", linestyle=(0, (4, 4)), linewidth=1.4),
        )
        ax.annotate(
            "",
            xy=(0.65, h),
            xytext=(d - 0.35, h),
            arrowprops=dict(arrowstyle="->", color=c, linewidth=3.0, shrinkA=0, shrinkB=0),
        )
        ax.text(d, h + 0.37, f"{h:g} m", ha="center", fontsize=12, color=c, fontweight="bold")

    ax.text(8.7, 5.85, "Esquema del protocolo de captura DroneFace", ha="center", fontsize=18, color="#111827", fontweight="bold")
    ax.text(8.7, 5.55, "Alturas 1.5, 3, 4 y 5 m; distancias de 2 a 17 m; imagenes raw con campo ultra amplio", ha="center", fontsize=11, color="#4b5563")
    ax.set_xlim(-0.8, 17.8)
    ax.set_ylim(-0.8, 6.25)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def make_dataset_summary_card() -> Path:
    out = OUT_DIR / "droneface_dataset_summary_card.png"
    width, height = 1500, 950
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    title = _font(64, bold=True)
    h2 = _font(35, bold=True)
    body = _font(34)
    small = _font(27)
    ink = (31, 41, 55)
    accent = (37, 99, 235)
    muted = (75, 85, 99)

    draw.text((80, 62), "DroneFace: resumen del dataset", font=title, fill=ink)
    draw.line((80, 150, 1420, 150), fill=(203, 213, 225), width=3)

    bullets = [
        ("11 sujetos", "7 hombres y 4 mujeres reportados en la descripcion original."),
        ("2057 imagenes", "620 imagenes raw, 1364 rostros frontales recortados y 73 retratos."),
        ("Captura con dron", "Resolucion 3680 x 2760, campo ultra amplio de 170 grados y luz diurna."),
        ("Variacion geometrica", "Alturas 1.5, 3, 4 y 5 m; distancias entre 2 y 17 m."),
        ("Uso en este proyecto", "Galerias desde trio_cam + trio_gp; evaluacion en photos_all_faces y photos_all."),
    ]

    y = 210
    for head, text in bullets:
        draw.rounded_rectangle((80, y + 6, 104, y + 30), radius=5, fill=accent)
        draw.text((130, y), head, font=h2, fill=ink)
        draw.text((130, y + 48), text, font=body if len(text) < 80 else small, fill=muted)
        y += 135

    draw.text((80, height - 84), "Figura generada desde la descripcion de DroneFace; no es una captura del paper.", font=small, fill=(100, 116, 139))
    canvas.save(out, quality=95)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = [
        make_pose_collage("trio_cam"),
        make_pose_collage("trio_gp"),
        make_capture_geometry(),
        make_dataset_summary_card(),
    ]
    for path in created:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
