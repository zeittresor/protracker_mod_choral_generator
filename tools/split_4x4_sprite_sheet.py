#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# github.com/zeittresor

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops
except ImportError:
    print("Pillow fehlt. Installiere es mit:")
    print("  pip install pillow")
    sys.exit(1)


def detect_content_bbox(img, threshold=8):
    rgba = img.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size

    min_x, min_y = w, h
    max_x, max_y = -1, -1

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > 0 and (r > threshold or g > threshold or b > threshold):
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x < min_x or max_y < min_y:
        return None

    return (min_x, min_y, max_x + 1, max_y + 1)


def split_sheet(sheet, cols, rows):
    w, h = sheet.size
    frame_w = w // cols
    frame_h = h // rows

    used_w = frame_w * cols
    used_h = frame_h * rows

    if used_w != w or used_h != h:
        print(f"Achtung: Bildgröße {w}x{h} ist nicht glatt durch {cols}x{rows} teilbar.")
        print(f"Es werden nur die ersten {used_w}x{used_h} Pixel verwendet.")

    frames = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * frame_w
            y1 = row * frame_h
            x2 = x1 + frame_w
            y2 = y1 + frame_h
            frames.append(sheet.crop((x1, y1, x2, y2)).convert("RGBA"))

    return frames, frame_w, frame_h


def stabilize_frames(frames, bg=(0, 0, 0, 255), threshold=8):
    bboxes = [detect_content_bbox(frame, threshold=threshold) for frame in frames]
    valid = [bbox for bbox in bboxes if bbox is not None]

    if not valid:
        return frames

    centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in valid]
    target_x = round(sum(c[0] for c in centers) / len(centers))
    target_y = round(sum(c[1] for c in centers) / len(centers))

    stabilized = []

    for frame, bbox in zip(frames, bboxes):
        if bbox is None:
            stabilized.append(frame)
            continue

        x1, y1, x2, y2 = bbox
        content = frame.crop(bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        dx = round(target_x - cx)
        dy = round(target_y - cy)

        canvas = Image.new("RGBA", frame.size, bg)
        canvas.alpha_composite(content, (x1 + dx, y1 + dy))
        stabilized.append(canvas)

    return stabilized


def make_pingpong(frames):
    if len(frames) <= 2:
        return frames
    return frames + frames[-2:0:-1]


def make_crossfade_frames(frames, steps=5):
    output = []
    for i in range(len(frames)):
        a = frames[i].convert("RGBA")
        b = frames[(i + 1) % len(frames)].convert("RGBA")
        output.append(a)
        for s in range(1, steps + 1):
            alpha = s / (steps + 1)
            output.append(Image.blend(a, b, alpha))
    return output


def save_frames(frames, out_dir, prefix, ext):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for idx, frame in enumerate(frames, start=1):
        path = out_dir / f"{prefix}{idx:02d}.{ext}"
        if ext.lower() in ("jpg", "jpeg"):
            frame.convert("RGB").save(path, quality=95)
        else:
            frame.save(path)
        saved.append(path)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Splitte ein 4x4-Spritesheet exakt in 16 Frames und optional stabilisiere die Figurposition."
    )
    parser.add_argument("input", help="Pfad zum Spritesheet, z.B. thinking_sheet.png")
    parser.add_argument("-o", "--out", default="split_frames", help="Ausgabeordner")
    parser.add_argument("--cols", type=int, default=4, help="Spaltenanzahl, Standard: 4")
    parser.add_argument("--rows", type=int, default=4, help="Zeilenanzahl, Standard: 4")
    parser.add_argument("--prefix", default="frame_", help="Dateiprefix, Standard: frame_")
    parser.add_argument("--ext", default="png", choices=["png", "webp", "jpg", "jpeg"], help="Ausgabeformat")
    parser.add_argument("--stabilize", action="store_true", help="Figur innerhalb jedes Frames anhand nicht-schwarzer Pixel zentrieren")
    parser.add_argument("--threshold", type=int, default=8, help="Schwarz-Schwelle für Stabilisierung, Standard: 8")
    parser.add_argument("--pingpong", action="store_true", help="Animation vorwärts/rückwärts ausgeben für weichere Loops")
    parser.add_argument("--gif", action="store_true", help="Zusätzlich preview.gif erzeugen")
    parser.add_argument("--crossfade", type=int, default=0, help="Anzahl Zwischenbilder pro Übergang für preview.gif, z.B. 5")
    parser.add_argument("--duration", type=int, default=110, help="GIF-Frame-Dauer in ms, Standard: 110")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.exists():
        print(f"Datei nicht gefunden: {input_path}")
        sys.exit(1)

    sheet = Image.open(input_path).convert("RGBA")
    frames, frame_w, frame_h = split_sheet(sheet, args.cols, args.rows)

    if args.stabilize:
        frames = stabilize_frames(frames, threshold=args.threshold)

    output_frames = make_pingpong(frames) if args.pingpong else frames
    saved = save_frames(output_frames, out_dir, args.prefix, args.ext)

    print(f"Gespeichert: {len(saved)} Frames in {out_dir}")
    print(f"Einzelframe-Größe: {frame_w}x{frame_h}px")

    if args.gif:
        gif_frames = output_frames
        if args.crossfade > 0:
            gif_frames = make_crossfade_frames(gif_frames, steps=args.crossfade)

        gif_path = out_dir / "preview.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.duration,
            loop=0,
            disposal=2,
        )
        print(f"Preview-GIF gespeichert: {gif_path}")


if __name__ == "__main__":
    main()
