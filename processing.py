from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np

ColorSpace = Literal["rgb"]


@dataclass(slots=True)
class PipelineConfig:
	global_method: Literal["fixed", "otsu", "triangle"] = "otsu"
	global_threshold: int = 128
	adaptive_block: int = 25
	adaptive_c: int = 5
	sharpen_method: Literal["unsharp", "laplacian"] = "unsharp"
	unsharp_amount: float = 1.5
	unsharp_radius: int = 3
	laplacian_ksize: int = 3


def load_image(path: Path) -> np.ndarray:
	image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Не удалось прочитать файл {path}")
	return image


def decode_image(file_bytes: bytes) -> np.ndarray:
	data = np.frombuffer(file_bytes, dtype=np.uint8)
	image = cv2.imdecode(data, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError("Файл не является поддерживаемым изображением.")
	return image


def to_gray(image: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_global_threshold(gray: np.ndarray, config: PipelineConfig) -> Tuple[np.ndarray, int]:
	if config.global_method == "fixed":
		threshold = int(np.clip(config.global_threshold, 0, 255))
		_, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
		return binary, threshold

	if config.global_method == "triangle":
		threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]
		return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1], int(threshold)

	threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
	return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1], int(threshold)


def apply_adaptive_threshold(gray: np.ndarray, config: PipelineConfig) -> np.ndarray:
	block = config.adaptive_block
	if block % 2 == 0:
		block += 1
	block = max(3, min(block, 99))
	c_value = max(0, min(config.adaptive_c, 50))
	return cv2.adaptiveThreshold(
		gray,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		block,
		c_value,
	)


def apply_sharpen(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
	if config.sharpen_method == "laplacian":
		k = config.laplacian_ksize
		if k % 2 == 0:
			k += 1
		k = max(1, min(k, 7))
		lap = cv2.Laplacian(image, cv2.CV_16S, ksize=k)
		sharp = cv2.convertScaleAbs(image - lap)
		return np.clip(sharp, 0, 255).astype(np.uint8)

	radius = max(1, min(config.unsharp_radius, 21))
	if radius % 2 == 0:
		radius += 1
	blurred = cv2.GaussianBlur(image, (radius, radius), 0)
	amount = np.clip(config.unsharp_amount, 0.0, 5.0)
	sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
	return np.clip(sharp, 0, 255).astype(np.uint8)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image
	return bgr_to_rgb(image)

