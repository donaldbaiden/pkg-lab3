from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DATASET_DIR = Path(__file__).parent / "test_images"


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def make_low_contrast(path: Path, size: Tuple[int, int]) -> None:
	array = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
	array[:, size[0] // 3 : 2 * size[0] // 3] = 135
	image = Image.fromarray(array, mode="RGB")
	draw = ImageDraw.Draw(image)
	draw.text((20, size[1] // 2 - 10), "low contrast", fill=(150, 150, 150))
	image.save(path)


def make_noisy(path: Path, size: Tuple[int, int]) -> None:
	base = np.full((size[1], size[0]), 128, dtype=np.uint8)
	noise = np.random.normal(0, 40, base.shape).astype(np.int16)
	noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
	Image.fromarray(noisy, mode="L").save(path)


def make_blurred(path: Path, size: Tuple[int, int]) -> None:
	array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
	for y in range(size[1]):
		color = int(255 * (y / size[1]))
		array[y, :, :] = (color, 255 - color, color // 2)
	image = Image.fromarray(array, mode="RGB").filter(ImageFilter.GaussianBlur(radius=8))
	draw = ImageDraw.Draw(image)
	draw.text((20, size[1] // 2 - 10), "blurred", fill=(255, 255, 255))
	image.save(path)


def make_shadowed(path: Path, size: Tuple[int, int]) -> None:
	array = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
	yy, xx = np.ogrid[: size[1], : size[0]]
	center = (size[1] / 2, size[0] / 2)
	dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
	mask = dist / dist.max()
	array = (array * (0.5 + 0.5 * mask[..., None])).astype(np.uint8)
	image = Image.fromarray(array, mode="RGB")
	draw = ImageDraw.Draw(image)
	draw.text((35, size[1] - 50), "shadow", fill=(255, 255, 255))
	image.save(path)


def main() -> None:
	ensure_dir(DATASET_DIR)
	make_low_contrast(DATASET_DIR / "low_contrast.png", (480, 320))
	make_noisy(DATASET_DIR / "noisy.png", (480, 320))
	make_blurred(DATASET_DIR / "blurred.png", (480, 320))
	make_shadowed(DATASET_DIR / "shadow.png", (480, 320))
	print(f"Созданы тестовые изображения в {DATASET_DIR}")


if __name__ == "__main__":
	main()

