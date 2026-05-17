#!/usr/bin/env python3
"""Generate a reference-based fixed-camera DEEPX can defect segmentation dataset.

The product image is fixed from a reference render. Each sample keeps product
size and position unchanged, varies lighting by +/-10%, and overlays known
defect masks that are converted to Ultralytics YOLO segmentation polygons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


IMAGE_SIZE = 1280
CLASS_NAMES = [
    "no_defect",
    "scratch",
    "dent",
    "stain",
]
DEFECT_CLASS_IDS = [1, 2, 3]

CLASS_COLORS = {
    0: (160, 160, 160),
    1: (247, 196, 83),
    2: (80, 180, 255),
    3: (190, 112, 255),
}

RESAMPLE_LANCZOS = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class CanGeometry:
    can_box: tuple[int, int, int, int]
    label_box: tuple[int, int, int, int]
    defect_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SegmentAnnotation:
    class_id: int
    class_name: str
    polygon: list[tuple[int, int]]
    bbox_xywh: list[int]
    area: int


@dataclass(frozen=True)
class ReferenceScene:
    name: str
    image: Image.Image
    mask: Image.Image
    geometry: CanGeometry


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    split: str
    defects: list[str]
    instance_count: int
    lighting_scale: float
    reference_name: str


def clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(value)))


def find_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def alpha_composite_clipped(base: Image.Image, overlay: Image.Image, clip_mask: Image.Image) -> None:
    clipped = overlay.copy()
    clipped.putalpha(ImageChops.multiply(clipped.getchannel("A"), clip_mask))
    base.alpha_composite(clipped)


def estimate_foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    raw = np.where((gray < 246) | (saturation > 18), 255, 0).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if component_count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[largest]
    return int(x), int(y), int(x + w), int(y + h)


def crop_square_around_product(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image

    side = min(width, height)
    bbox = estimate_foreground_bbox(image)
    if bbox is None:
        center_x = width // 2
        center_y = height // 2
    else:
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

    left = clamp(center_x - side // 2, 0, width - side)
    top = clamp(center_y - side // 2, 0, height - side)
    return image.crop((left, top, left + side, top + side))


def normalize_reference_image(path: Path, size: int = IMAGE_SIZE) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")
    image = Image.open(path).convert("RGB")
    image = crop_square_around_product(image)
    if image.size != (size, size):
        image = image.resize((size, size), RESAMPLE_LANCZOS)
    return image


def infer_reference_product_mask(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    arr = np.asarray(image).astype(np.int16)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)

    raw = np.where((gray < 246) | (saturation > 18), 255, 0).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if component_count <= 1:
        raise RuntimeError("Could not infer product mask from reference image")

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(filled)

    # The reference has a soft cast shadow under the can. It is useful in the
    # base image, but should not be treated as defectable product surface.
    filled[y + round(h * 0.94) :, :] = 0
    x, y, w, h = cv2.boundingRect(filled)
    return Image.fromarray(filled, mode="L").filter(ImageFilter.GaussianBlur(radius=0.35)), (x, y, x + w, y + h)


def build_reference_geometry(product_box: tuple[int, int, int, int]) -> CanGeometry:
    x0, y0, x1, y1 = product_box
    width = x1 - x0
    height = y1 - y0

    label_box = (
        x0 + round(width * 0.04),
        y0 + round(height * 0.19),
        x1 - round(width * 0.06),
        y1 - round(height * 0.07),
    )
    defect_box = (
        x0 + round(width * 0.08),
        y0 + round(height * 0.23),
        x1 - round(width * 0.13),
        y0 + round(height * 0.86),
    )
    return CanGeometry(product_box, label_box, defect_box)


def load_reference_scene(reference_path: Path) -> ReferenceScene:
    image = normalize_reference_image(reference_path).convert("RGBA")
    product_mask, product_box = infer_reference_product_mask(image.convert("RGB"))
    geometry = build_reference_geometry(product_box)
    return ReferenceScene(reference_path.name, image, product_mask, geometry)


def paste_with_offset(canvas: Image.Image, source: Image.Image, offset: tuple[int, int]) -> None:
    x, y = offset
    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    width = min(source.width - src_x0, canvas.width - dst_x0)
    height = min(source.height - src_y0, canvas.height - dst_y0)
    if width <= 0 or height <= 0:
        return
    crop = source.crop((src_x0, src_y0, src_x0 + width, src_y0 + height))
    canvas.paste(crop, (dst_x0, dst_y0), crop if crop.mode == "RGBA" else None)


def align_reference_scene(scene: ReferenceScene, target_box: tuple[int, int, int, int]) -> ReferenceScene:
    source_box = scene.geometry.can_box
    if source_box == target_box:
        return scene

    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    source_h = sy1 - sy0
    target_h = ty1 - ty0
    scale = target_h / max(source_h, 1)

    new_size = (round(IMAGE_SIZE * scale), round(IMAGE_SIZE * scale))
    resized_image = scene.image.resize(new_size, RESAMPLE_LANCZOS)
    resized_mask = scene.mask.resize(new_size, RESAMPLE_LANCZOS)

    source_cx = (sx0 + sx1) / 2 * scale
    source_cy = (sy0 + sy1) / 2 * scale
    target_cx = (tx0 + tx1) / 2
    target_cy = (ty0 + ty1) / 2
    offset = (round(target_cx - source_cx), round(target_cy - source_cy))

    image_canvas = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255, 255))
    mask_canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    paste_with_offset(image_canvas, resized_image, offset)
    paste_with_offset(mask_canvas, resized_mask, offset)

    geometry = build_reference_geometry(target_box)
    return ReferenceScene(scene.name, image_canvas, mask_canvas, geometry)


def load_reference_scenes(reference_paths: list[Path]) -> list[ReferenceScene]:
    scenes = [load_reference_scene(path) for path in reference_paths]
    if not scenes:
        raise ValueError("At least one reference image is required")

    target_box = scenes[0].geometry.can_box
    return [align_reference_scene(scene, target_box) for scene in scenes]


def random_blob_points(
    rng: random.Random,
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    points: int,
    jitter: float = 0.35,
) -> list[tuple[int, int]]:
    cx, cy = center
    phase = rng.uniform(0, math.tau)
    out = []
    for index in range(points):
        theta = phase + math.tau * index / points
        scale = rng.uniform(1.0 - jitter, 1.0 + jitter)
        out.append((round(cx + math.cos(theta) * radius_x * scale), round(cy + math.sin(theta) * radius_y * scale)))
    return out


def sample_defect_point(
    rng: random.Random,
    defect_box: tuple[int, int, int, int],
    anchor: tuple[int, int] | None = None,
) -> tuple[int, int]:
    x0, y0, x1, y1 = defect_box
    if anchor is None:
        return rng.randint(x0, x1), rng.randint(y0, y1)

    x = round(anchor[0] + rng.gauss(0, 54))
    y = round(anchor[1] + rng.gauss(0, 42))
    return clamp(x, x0, x1), clamp(y, y0, y1)


def add_scratch(
    scene: Image.Image,
    can_mask: Image.Image,
    geometry: CanGeometry,
    rng: random.Random,
    anchor: tuple[int, int] | None,
) -> Image.Image:
    mask = Image.new("L", scene.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    overlay = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cx, cy = sample_defect_point(rng, geometry.defect_box, anchor)
    angle = rng.uniform(-0.65, 0.65)
    scratch_palette = [
        (235, 238, 232, 220),
        (185, 194, 205, 205),
        (125, 135, 146, 190),
        (225, 190, 118, 190),
        (48, 57, 68, 165),
    ]
    count = rng.randint(2, 6)

    for index in range(count):
        local_angle = angle + rng.uniform(-0.12, 0.12)
        length = rng.randint(55, 240) if index else rng.randint(140, 330)
        width = rng.choices([1, 2, 3, 4, 5, 6], weights=[2, 4, 5, 4, 2, 1], k=1)[0]
        ox = rng.randint(-18, 18)
        oy = rng.randint(-12, 12)

        points: list[tuple[int, int]] = []
        steps = rng.randint(5, 10)
        for step in range(steps):
            t = step / (steps - 1) - 0.5
            along = t * length
            perpendicular = rng.gauss(0, width * 1.2 + 1.8)
            px = cx + ox + math.cos(local_angle) * along - math.sin(local_angle) * perpendicular
            py = cy + oy + math.sin(local_angle) * along + math.cos(local_angle) * perpendicular
            points.append((round(px), round(py)))

        color = rng.choice(scratch_palette)
        shadow_alpha = rng.randint(35, 95)
        shadow_offset = rng.choice([(1, 2), (2, 2), (-1, 2), (1, -1)])
        shadow_points = [(x + shadow_offset[0], y + shadow_offset[1]) for x, y in points]

        mask_draw.line(points, fill=255, width=width + 3, joint="curve")
        draw.line(shadow_points, fill=(0, 0, 0, shadow_alpha), width=width + 3, joint="curve")
        draw.line(points, fill=color, width=width, joint="curve")

        if rng.random() < 0.72:
            highlight_points = [(x - 1, y - 1) for x, y in points]
            draw.line(highlight_points, fill=(255, 255, 250, rng.randint(45, 130)), width=max(1, width // 2), joint="curve")

        if rng.random() < 0.45:
            branch_start = rng.choice(points[1:-1])
            branch_angle = local_angle + rng.choice([-1, 1]) * rng.uniform(0.35, 0.95)
            branch_len = rng.randint(20, 70)
            branch_end = (
                round(branch_start[0] + math.cos(branch_angle) * branch_len),
                round(branch_start[1] + math.sin(branch_angle) * branch_len),
            )
            branch_width = max(1, width - rng.randint(1, 3))
            mask_draw.line((branch_start, branch_end), fill=255, width=branch_width + 2)
            draw.line((branch_start, branch_end), fill=color[:3] + (rng.randint(100, 190),), width=branch_width)

    for _ in range(rng.randint(3, 9)):
        x0, y0 = sample_defect_point(rng, geometry.defect_box, anchor)
        hair_angle = angle + rng.uniform(-0.9, 0.9)
        hair_len = rng.randint(18, 82)
        p1 = (round(x0 - math.cos(hair_angle) * hair_len / 2), round(y0 - math.sin(hair_angle) * hair_len / 2))
        p2 = (round(x0 + math.cos(hair_angle) * hair_len / 2), round(y0 + math.sin(hair_angle) * hair_len / 2))
        hair_color = rng.choice(scratch_palette)
        mask_draw.line((p1, p2), fill=255, width=2)
        draw.line((p1, p2), fill=hair_color[:3] + (rng.randint(70, 145),), width=1)

    alpha_composite_clipped(scene, overlay.filter(ImageFilter.GaussianBlur(radius=0.18)), can_mask)
    return ImageChops.multiply(mask, can_mask)


def add_dent(
    scene: Image.Image,
    can_mask: Image.Image,
    geometry: CanGeometry,
    rng: random.Random,
    anchor: tuple[int, int] | None,
) -> Image.Image:
    cx, cy = sample_defect_point(rng, geometry.defect_box, anchor)
    rx = rng.randint(58, 118)
    ry = rng.randint(38, 82)
    angle = rng.uniform(-0.35, 0.35)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    x_rel = xx - cx
    y_rel = yy - cy
    xr = x_rel * cos_a + y_rel * sin_a
    yr = -x_rel * sin_a + y_rel * cos_a
    r2 = (xr / rx) ** 2 + (yr / ry) ** 2
    soft = np.clip(1.0 - r2, 0.0, 1.0) ** 1.55
    rim = np.exp(-((np.sqrt(np.maximum(r2, 1e-6)) - 0.93) ** 2) / 0.004)
    light_dir = (-0.72 * xr / rx) + (-0.46 * yr / ry)

    binary = (r2 <= 1.02).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")
    mask = ImageChops.multiply(mask, can_mask)

    shadow_alpha = np.clip(soft * 62 + rim * np.maximum(-light_dir, 0) * 72, 0, 118).astype(np.uint8)
    shadow = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(shadow_alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=2.2)))
    alpha_composite_clipped(scene, shadow, can_mask)

    highlight_alpha = np.clip(rim * np.maximum(light_dir, 0) * 96 + soft * np.maximum(light_dir, 0) * 18, 0, 120).astype(np.uint8)
    highlight = Image.new("RGBA", scene.size, (210, 232, 255, 0))
    highlight.putalpha(Image.fromarray(highlight_alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=1.4)))
    alpha_composite_clipped(scene, highlight, can_mask)

    detail = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    detail_draw = ImageDraw.Draw(detail)
    for _ in range(rng.randint(2, 4)):
        crease_len = rng.randint(int(rx * 0.45), int(rx * 0.95))
        crease_y = cy + rng.randint(-ry // 4, ry // 4)
        crease_x = cx + rng.randint(-rx // 5, rx // 5)
        points = []
        for step in range(7):
            t = (step / 6 - 0.5) * crease_len
            wobble = math.sin(step * 1.7 + rng.random()) * rng.uniform(1.8, 4.5)
            px = crease_x + math.cos(angle) * t - math.sin(angle) * wobble
            py = crease_y + math.sin(angle) * t + math.cos(angle) * wobble
            points.append((round(px), round(py)))
        detail_draw.line(points, fill=(5, 12, 22, rng.randint(45, 75)), width=rng.randint(2, 4), joint="curve")
        shifted = [(x - 1, y - 2) for x, y in points]
        detail_draw.line(shifted, fill=(215, 235, 255, rng.randint(35, 62)), width=1, joint="curve")

    alpha_composite_clipped(scene, detail.filter(ImageFilter.GaussianBlur(radius=0.45)), mask)
    return ImageChops.multiply(mask, can_mask)


def add_stain(
    scene: Image.Image,
    can_mask: Image.Image,
    geometry: CanGeometry,
    rng: random.Random,
    anchor: tuple[int, int] | None,
) -> Image.Image:
    mask = Image.new("L", scene.size, 0)
    cx, cy = sample_defect_point(rng, geometry.defect_box, anchor)
    mask_draw = ImageDraw.Draw(mask)
    stain_palette = [
        ((72, 42, 17), (105, 170)),      # coffee/oil brown
        ((24, 29, 34), (75, 140)),       # dark grease
        ((188, 192, 178), (58, 112)),    # dried mineral residue
        ((39, 82, 128), (52, 118)),      # blue cleaner mark
        ((142, 96, 34), (65, 135)),      # amber sticky residue
        ((92, 38, 92), (48, 105)),       # faint chemical tint
    ]
    base_color, alpha_range = rng.choice(stain_palette)

    rx = rng.randint(36, 112)
    ry = rng.randint(26, 98)
    points = random_blob_points(rng, (cx, cy), rx, ry, rng.randint(9, 19), jitter=rng.uniform(0.35, 0.72))
    mask_draw.polygon(points, fill=255)

    for _ in range(rng.randint(0, 4)):
        sx = cx + rng.randint(-rx, rx)
        sy = cy + rng.randint(-ry, ry)
        sr = rng.randint(6, 22)
        mask_draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=255)

    if rng.random() < 0.7:
        for _ in range(rng.randint(1, 3)):
            drip_x = cx + rng.randint(-rx // 3, rx // 3)
            drip_len = rng.randint(28, 125)
            drip_w = rng.randint(5, 15)
            start_y = cy + rng.randint(0, max(4, ry // 2))
            mask_draw.rounded_rectangle(
                (drip_x - drip_w, start_y, drip_x + drip_w, start_y + drip_len),
                radius=drip_w,
                fill=255,
            )

    if rng.random() < 0.35:
        hole_rx = max(8, rx // rng.randint(3, 5))
        hole_ry = max(8, ry // rng.randint(3, 5))
        mask_draw.ellipse((cx - hole_rx, cy - hole_ry, cx + hole_rx, cy + hole_ry), fill=0)

    mask_arr = np.array(mask.point(lambda value: 255 if value > 0 else 0), dtype=np.uint8)
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    texture = noise_rng.normal(1.0, rng.uniform(0.08, 0.24), mask_arr.shape)
    opacity = rng.randint(*alpha_range) / 255.0
    alpha_arr = np.clip(mask_arr.astype(np.float32) * opacity * texture, 0, 180).astype(np.uint8)
    alpha = Image.fromarray(alpha_arr, mode="L").filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.5, 8.0)))
    stain = Image.new("RGBA", scene.size, base_color + (0,))
    stain.putalpha(alpha)

    edge = cv2.morphologyEx(mask_arr, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    edge_alpha = Image.fromarray(np.clip(edge * rng.uniform(0.18, 0.45), 0, 115).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )
    edge_color = tuple(clamp(channel * rng.uniform(0.62, 0.9)) for channel in base_color)
    edge_layer = Image.new("RGBA", scene.size, edge_color + (0,))
    edge_layer.putalpha(edge_alpha)
    stain.alpha_composite(edge_layer)

    gloss = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(gloss)
    for _ in range(rng.randint(1, 4)):
        gx0 = cx + rng.randint(-rx // 2, rx // 2)
        gy0 = cy + rng.randint(-ry // 2, ry // 3)
        gx1 = gx0 + rng.randint(-28, 70)
        gy1 = gy0 + rng.randint(-12, 22)
        draw.line((gx0, gy0, gx1, gy1), fill=(255, 255, 244, rng.randint(22, 62)), width=rng.randint(2, 5))
    stain.alpha_composite(gloss.filter(ImageFilter.GaussianBlur(radius=3)))

    alpha_composite_clipped(scene, stain, can_mask)
    return ImageChops.multiply(mask, can_mask)


DefectRenderer = Callable[[Image.Image, Image.Image, CanGeometry, random.Random, tuple[int, int] | None], Image.Image]

DEFECT_RENDERERS: dict[int, DefectRenderer] = {
    1: add_scratch,
    2: add_dent,
    3: add_stain,
}


def mask_to_segment_annotations(mask: Image.Image, class_id: int, min_area: int = 80) -> list[SegmentAnnotation]:
    arr = np.array(mask.point(lambda value: 255 if value > 0 else 0), dtype=np.uint8)
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations: list[SegmentAnnotation] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.5, perimeter * 0.004)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]

        polygon = [(clamp(x, 0, IMAGE_SIZE - 1), clamp(y, 0, IMAGE_SIZE - 1)) for x, y in polygon]
        if len(set(polygon)) < 3:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        annotations.append(
            SegmentAnnotation(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                polygon=polygon,
                bbox_xywh=[int(x), int(y), int(w), int(h)],
                area=area,
            )
        )

    return annotations


def apply_lighting_variation(image: Image.Image, lighting_scale: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    luminance_noise = noise_rng.normal(0, 0.55, arr.shape[:2])[..., None]
    arr = arr * lighting_scale + luminance_noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def render_sample(
    base_scene: Image.Image,
    can_mask: Image.Image,
    geometry: CanGeometry,
    rng: random.Random,
    class_ids: list[int],
    overlap: bool,
    lighting_scale: float,
) -> tuple[Image.Image, Image.Image, list[SegmentAnnotation]]:
    scene = base_scene.copy()
    semantic_mask = Image.new("L", scene.size, 0)
    annotations: list[SegmentAnnotation] = []

    anchor = sample_defect_point(rng, geometry.defect_box) if overlap and len(class_ids) > 1 else None

    for class_id in class_ids:
        defect_mask = DEFECT_RENDERERS[class_id](scene, can_mask, geometry, rng, anchor)
        defect_mask = defect_mask.point(lambda value: 255 if value > 0 else 0)
        class_mask = Image.new("L", scene.size, class_id)
        semantic_mask.paste(class_mask, (0, 0), defect_mask)
        annotations.extend(mask_to_segment_annotations(defect_mask, class_id))

    return apply_lighting_variation(scene.convert("RGB"), lighting_scale, rng), semantic_mask, annotations


def choose_defect_plan(
    count: int,
    normal_ratio: float,
    max_defects: int,
    overlap_ratio: float,
    rng: random.Random,
) -> list[tuple[list[int], bool]]:
    normal_count = round(count * normal_ratio)
    defect_count = count - normal_count
    plan: list[tuple[list[int], bool]] = [([], False) for _ in range(normal_count)]

    for index in range(defect_count):
        primary = DEFECT_CLASS_IDS[index % len(DEFECT_CLASS_IDS)]
        if max_defects == 1:
            defect_total = 1
        else:
            defect_total = rng.choices([1, 2, 3], weights=[0.48, 0.34, 0.18], k=1)[0]
            defect_total = min(defect_total, max_defects)

        candidates = [class_id for class_id in DEFECT_CLASS_IDS if class_id != primary]
        rng.shuffle(candidates)
        class_ids = [primary] + candidates[: defect_total - 1]
        rng.shuffle(class_ids)
        overlap = len(class_ids) > 1 and rng.random() < overlap_ratio
        plan.append((class_ids, overlap))

    rng.shuffle(plan)
    return plan


def build_split_plan(count: int, train_ratio: float, val_ratio: float, rng: random.Random) -> list[str]:
    train_count = round(count * train_ratio)
    val_count = round(count * val_ratio)
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = count - train_count - val_count
    splits = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
    rng.shuffle(splits)
    return splits


def prepare_output_dirs(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "masks" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)


def yolo_segment_line(annotation: SegmentAnnotation) -> str:
    coords: list[str] = []
    for x, y in annotation.polygon:
        coords.append(f"{x / IMAGE_SIZE:.6f}")
        coords.append(f"{y / IMAGE_SIZE:.6f}")
    return f"{annotation.class_id} {' '.join(coords)}"


def write_yolo_segment_label(label_path: Path, annotations: list[SegmentAnnotation]) -> None:
    lines = [yolo_segment_line(annotation) for annotation in annotations]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(out_dir: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    text = f"""# Synthetic fixed-camera DEEPX can defect segmentation dataset.
# ID 0 is no_defect. Normal images intentionally have empty YOLO label files.
path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test
names:
{names}
"""
    (out_dir / "dataset.yaml").write_text(text, encoding="utf-8")


def write_metadata(out_dir: Path, records: list[ImageRecord]) -> None:
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image_id",
                "file_name",
                "split",
                "image_class_id",
                "image_class_name",
                "defects",
                "instance_count",
                "lighting_scale",
                "reference_name",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "file_name": record.file_name,
                    "split": record.split,
                    "image_class_id": 0 if not record.defects else 1,
                    "image_class_name": "no_defect" if not record.defects else "defective",
                    "defects": "|".join(record.defects),
                    "instance_count": record.instance_count,
                    "lighting_scale": f"{record.lighting_scale:.4f}",
                    "reference_name": record.reference_name,
                }
            )


def write_coco(out_dir: Path, split: str, images: list[dict], annotations: list[dict]) -> None:
    categories = [{"id": index, "name": name, "supercategory": "deepx_can_surface"} for index, name in enumerate(CLASS_NAMES)]
    payload = {"images": images, "annotations": annotations, "categories": categories}
    (out_dir / "annotations" / f"coco_{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_preview(out_dir: Path, records: list[ImageRecord], max_items: int = 20) -> None:
    if not records:
        return

    selected = records[:max_items]
    cell = 256
    label_h = 30
    columns = min(5, len(selected))
    rows = math.ceil(len(selected) / columns)
    preview = Image.new("RGB", (columns * cell, rows * (cell + label_h)), (236, 236, 236))
    caption_font = find_font(12, bold=True)
    tag_font = find_font(11, bold=True)

    for idx, record in enumerate(selected):
        image_path = out_dir / record.file_name
        image = Image.open(image_path).convert("RGB").resize((cell, cell), RESAMPLE_LANCZOS)
        draw = ImageDraw.Draw(image, "RGBA")
        label_path = out_dir / "labels" / record.split / (Path(record.file_name).stem + ".txt")

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                points = [(coords[i] * cell, coords[i + 1] * cell) for i in range(0, len(coords), 2)]
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                draw.polygon(points, outline=color + (255,), fill=color + (40,))
                min_x = min(x for x, _ in points)
                min_y = min(y for _, y in points)
                text = CLASS_NAMES[class_id]
                text_bbox = draw.textbbox((0, 0), text, font=tag_font)
                draw.rectangle((min_x, max(0, min_y - 17), min_x + text_bbox[2] + 8, max(16, min_y)), fill=color + (235,))
                draw.text((min_x + 4, max(0, min_y - 16)), text, fill=(20, 20, 20), font=tag_font)

        col = idx % columns
        row = idx // columns
        x = col * cell
        y = row * (cell + label_h)
        preview.paste(image, (x, y))
        caption = "no_defect" if not record.defects else ", ".join(record.defects)
        ImageDraw.Draw(preview).text((x + 6, y + cell + 7), caption[:38], fill=(30, 30, 30), font=caption_font)

    preview.save(out_dir / "preview_grid.jpg", quality=92)


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_dir = args.out
    prepare_output_dirs(out_dir, args.clean)

    reference_scenes = load_reference_scenes(args.reference_images)
    plan = choose_defect_plan(args.count, args.normal_ratio, args.max_defects, args.overlap_ratio, rng)
    splits = build_split_plan(args.count, args.train_ratio, args.val_ratio, rng)

    records: list[ImageRecord] = []
    coco_images: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    coco_annotations: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    annotation_id = 1

    for index, (class_ids, overlap) in enumerate(plan, start=1):
        split = splits[index - 1]
        lighting_scale = rng.uniform(0.9, 1.1)
        reference_scene = reference_scenes[(index - 1) % len(reference_scenes)]
        image, semantic_mask, annotations = render_sample(
            reference_scene.image,
            reference_scene.mask,
            reference_scene.geometry,
            rng,
            class_ids,
            overlap,
            lighting_scale,
        )

        stem = f"deepx_can_{index:06d}"
        image_rel = Path("images") / split / f"{stem}.png"
        label_path = out_dir / "labels" / split / f"{stem}.txt"
        mask_path = out_dir / "masks" / split / f"{stem}.png"

        image.save(out_dir / image_rel)
        semantic_mask.save(mask_path)
        write_yolo_segment_label(label_path, annotations)

        defects = sorted({annotation.class_name for annotation in annotations})
        records.append(ImageRecord(index, str(image_rel), split, defects, len(annotations), lighting_scale, reference_scene.name))
        coco_images[split].append({"id": index, "file_name": str(image_rel), "width": IMAGE_SIZE, "height": IMAGE_SIZE})

        for annotation in annotations:
            segmentation = [[coord for point in annotation.polygon for coord in point]]
            coco_annotations[split].append(
                {
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": annotation.class_id,
                    "bbox": annotation.bbox_xywh,
                    "area": annotation.area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    for split in ("train", "val", "test"):
        write_coco(out_dir, split, coco_images[split], coco_annotations[split])
    write_dataset_yaml(out_dir)
    write_metadata(out_dir, records)
    draw_preview(out_dir, records)

    print(f"Wrote {len(records)} reference-based fixed-camera 1280x1280 images to {out_dir}")
    print(f"Classes: {', '.join(f'{i}:{name}' for i, name in enumerate(CLASS_NAMES))}")
    print(f"References: {', '.join(scene.name for scene in reference_scenes)}")
    print("Lighting scale: 0.90-1.10")
    print(f"YOLO segmentation labels: {out_dir / 'labels'}")
    print(f"Preview: {out_dir / 'preview_grid.jpg'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_deepx_can_seg"), help="Output dataset directory.")
    parser.add_argument(
        "--reference-image",
        "--reference-images",
        dest="reference_images",
        type=Path,
        nargs="+",
        default=None,
        help="One or more fixed good-product reference images used as normal sample bases.",
    )
    parser.add_argument("--count", type=int, default=200, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible defects.")
    parser.add_argument("--normal-ratio", type=float, default=0.15, help="Fraction of no_defect samples.")
    parser.add_argument("--max-defects", type=int, default=3, choices=(1, 2, 3), help="Maximum defect types per image.")
    parser.add_argument("--overlap-ratio", type=float, default=0.35, help="Chance that multi-defect images share a local overlap area.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio. Test receives the remainder.")
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before generation.")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.reference_images is None:
        default_references = sorted(Path("assets").glob("deepx_can_reference_*.png"))
        if not default_references:
            default_references = [Path("assets/deepx_can_reference.png")]
        args.reference_images = default_references
    missing = [path for path in args.reference_images if not path.exists()]
    if missing:
        parser.error("missing reference image(s): " + ", ".join(str(path) for path in missing))
    if not 0 <= args.normal_ratio < 1:
        parser.error("--normal-ratio must be in [0, 1)")
    if not 0 <= args.overlap_ratio <= 1:
        parser.error("--overlap-ratio must be in [0, 1]")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be in (0, 1)")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("--train-ratio + --val-ratio must be less than 1")
    return args


if __name__ == "__main__":
    generate_dataset(parse_args())
