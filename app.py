from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from PIL import Image

from processing import (
	PipelineConfig,
	apply_adaptive_threshold,
	apply_global_threshold,
	apply_sharpen,
	decode_image,
	load_image,
	normalize_for_display,
	to_gray,
)

st.set_page_config(page_title="Lab3 · Пороговая обработка и резкость", layout="wide")
st.title("Lab3 · Пороговая обработка и повышение резкости")

DATASET_DIR = Path(__file__).parent / "test_images"


def list_dataset() -> List[Path]:
	if not DATASET_DIR.exists():
		return []
	return sorted(p for p in DATASET_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def load_from_uploader() -> np.ndarray | None:
	file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg", "bmp"])
	if file:
		return decode_image(file.read())
	return None


def select_sample(samples: List[Path]) -> np.ndarray | None:
	if not samples:
		st.warning("Папка test_images пока пуста. Запустите generate_test_images.py.")
		return None
	filename = st.selectbox("Или выберите образец из набора", samples, format_func=lambda p: p.name)
	if filename:
		return load_image(filename)
	return None


def to_pil(image: np.ndarray) -> Image.Image:
	return Image.fromarray(normalize_for_display(image))


def main() -> None:
	samples = list_dataset()
	st.sidebar.header("Параметры pipeline")

	config = PipelineConfig()
	config.global_method = st.sidebar.selectbox(
		"Глобальный метод",
		options=["otsu", "triangle", "fixed"],
		format_func=lambda x: {"otsu": "Оцу", "triangle": "Треугольник", "fixed": "Фиксированный"}[x],
	)
	if config.global_method == "fixed":
		config.global_threshold = st.sidebar.slider("Порог (0-255)", 0, 255, 128)

	config.adaptive_block = st.sidebar.slider("Окно адаптивного порога (нечётное)", 3, 99, 25, step=2)
	config.adaptive_c = st.sidebar.slider("Коррекция C", 0, 30, 5)

	config.sharpen_method = st.sidebar.selectbox(
		"Метод повышения резкости",
		options=["unsharp", "laplacian"],
		format_func=lambda x: {"unsharp": "Unsharp mask", "laplacian": "Лапласиан"}[x],
	)
	if config.sharpen_method == "unsharp":
		config.unsharp_radius = st.sidebar.slider("Радиус blur", 1, 21, 5, step=2)
		config.unsharp_amount = st.sidebar.slider("Коэффициент", 0.0, 3.0, 1.5, step=0.1)
	else:
		config.laplacian_ksize = st.sidebar.slider("Размер ядра Лапласа", 1, 7, 3, step=2)

	st.sidebar.markdown("---")
	st.sidebar.caption("Доступные образцы:")
	for sample in samples:
		st.sidebar.write(f"- {sample.name}")

	st.markdown("### Выберите изображение")
	image = load_from_uploader()
	if image is None:
		image = select_sample(samples)

	if image is None:
		st.info("Загрузите файл или создайте набор test_images.")
		return

	gray = to_gray(image)
	global_binary, selected_threshold = apply_global_threshold(gray, config)
	adaptive_binary = apply_adaptive_threshold(gray, config)
	sharpened = apply_sharpen(image, config)

	col_original, col_sharp = st.columns(2)
	with col_original:
		st.markdown("**Оригинал**")
		st.image(to_pil(image), use_column_width=True)
	with col_sharp:
		st.markdown("**Высокочастотная фильтрация**")
		st.image(to_pil(sharpened), use_column_width=True)

	col_gray, col_hist = st.columns(2)
	with col_gray:
		st.markdown("**Градации серого**")
		st.image(gray, clamp=True, use_column_width=True)
	with col_hist:
		hist_values, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
		st.bar_chart(hist_values)
		st.caption("Гистограмма оттенков (до обработки).")

	col_global, col_adaptive = st.columns(2)
	with col_global:
		st.markdown(f"**Глобальная пороговая обработка** (порог {selected_threshold})")
		st.image(global_binary, clamp=True, use_column_width=True)
	with col_adaptive:
		st.markdown("**Адаптивная пороговая обработка**")
		st.image(adaptive_binary, clamp=True, use_column_width=True)

	st.markdown("### Экспорт")
	st.download_button("Скачать sharpened.png", data=image_to_bytes(sharpened), file_name="sharpened.png")
	st.download_button("Скачать global_threshold.png", data=image_to_bytes(global_binary), file_name="global_threshold.png")
	st.download_button("Скачать adaptive_threshold.png", data=image_to_bytes(adaptive_binary), file_name="adaptive_threshold.png")


def image_to_bytes(image: np.ndarray) -> bytes:
	buffer = io.BytesIO()
	Image.fromarray(normalize_for_display(image)).save(buffer, format="PNG")
	return buffer.getvalue()


if __name__ == "__main__":
	main()

