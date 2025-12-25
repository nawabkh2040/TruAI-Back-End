"""Sightengine AI-generation detector helper.

Provides `detect_ai_generated(image, api_user, api_secret, ...)` which accepts
either a public image URL or a local file path and returns a tuple
(is_ai_generated: bool, score: float, raw_response: dict).

Requires the `requests` package.
"""

from typing import Tuple, Union, Optional
import requests
import os
import io
import tempfile

try:
	import PIL.Image as Image
except Exception:
 	Image = None


def _extract_score(resp: dict) -> Optional[float]:
	# Expecting response like {"type": {"ai_generated": 0.01}} per example
	if not isinstance(resp, dict):
		return None
	if "type" in resp and isinstance(resp["type"], dict):
		val = resp["type"].get("ai_generated")
		if isinstance(val, (int, float)):
			return float(val)
	# fallback: search top-level
	val = resp.get("ai_generated")
	if isinstance(val, (int, float)):
		return float(val)
	# no score found
	return None


def prepare_image_for_upload(src_path: str, max_bytes: int = 12 * 1024 * 1024, *, initial_quality: int = 85) -> Tuple[str, bool]:
	"""Return a file path ready for upload and a boolean indicating whether it's a temp file.

	If the source file is already <= `max_bytes` return `(src_path, False)`.
	Otherwise convert to WebP and try progressively reducing quality and size until under the limit.

	Requires Pillow (`pip install pillow`).
	"""

	if not os.path.isfile(src_path):
		raise ValueError(f"Local file not found: {src_path}")

	orig_size = os.path.getsize(src_path)
	if orig_size <= max_bytes:
		return src_path, False

	if Image is None:
		raise ValueError("Pillow is required to convert images to WebP (pip install pillow)")

	# Load image once
	img = Image.open(src_path)

	if img.mode not in ("RGB", "RGBA"):
		img = img.convert("RGB")

	quality = initial_quality
	scale = 1.0
	buf = None

	# Try several attempts: lower quality first, then downscale
	for attempt in range(12):
		w = max(1, int(img.width * scale))
		h = max(1, int(img.height * scale))
		if scale < 1.0:
			resized = img.resize((w, h), Image.LANCZOS)
		else:
			resized = img

		buf = io.BytesIO()
		try:
			resized.save(buf, format="WEBP", quality=int(quality), method=6)
		except Exception:
			# fallback to default save parameters
			buf = io.BytesIO()
			resized.save(buf, format="WEBP")

		size = buf.tell()
		if size <= max_bytes:
			# write to a temp file and return
			tf = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
			with open(tf.name, "wb") as f:
				f.write(buf.getvalue())
			return tf.name, True

		# adjust parameters
		if quality > 30:
			quality = max(20, quality - 20)
		else:
			scale = scale * 0.8

	# Final attempt: if we produced a buffer, write it; otherwise copy original
	if buf is None:
		# no conversion succeeded; copy original to a temp file
		tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(src_path)[1] or ".img")
		with open(src_path, "rb") as sf, open(tf.name, "wb") as df:
			df.write(sf.read())
		return tf.name, True

	tf = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
	with open(tf.name, "wb") as f:
		f.write(buf.getvalue())
	return tf.name, True


def detect_ai_generated(image: str, api_user: str, api_secret: str, *, models: str = "genai", threshold: float = 0.5, timeout: int = 15) -> Tuple[bool, float, dict]:
	"""Detect whether `image` is AI-generated using Sightengine API.

	Args:
		image: public image URL or local file path.
		api_user: Sightengine `api_user` credential.
		api_secret: Sightengine `api_secret` credential.
		models: models param for the API (default: "genai").
		threshold: score threshold above which image is considered AI-generated.
		timeout: request timeout in seconds.

	Returns:
		(is_ai_generated, score, raw_response)

	Raises:
		requests.RequestException on network errors.
		ValueError if API returns unexpected payload.
	"""
	url = "https://api.sightengine.com/1.0/check.json"
	params = {"models": models, "api_user": api_user, "api_secret": api_secret}

	# Determine whether image is a URL or file path
	is_url = isinstance(image, str) and image.lower().startswith(("http://", "https://"))

	try:
		if is_url:
			params.update({"url": image})
			resp = requests.get(url, params=params, timeout=timeout)
		else:
			# ensure file exists
			if not os.path.isfile(image):
				raise ValueError(f"Local file not found: {image}")
			# prepare a possibly smaller file only if needed
			upload_path = image
			is_temp = False
			try:
				upload_path, is_temp = prepare_image_for_upload(image, max_bytes=12 * 1024 * 1024)
			except Exception:
				# If conversion/prep fails, fall back to original file and let API return an error
				upload_path, is_temp = image, False

			with open(upload_path, "rb") as fh:
				files = {"media": fh}
				resp = requests.post(url, files=files, data=params, timeout=timeout)

		resp.raise_for_status()
		data = resp.json()

		score = _extract_score(data)
		if score is None:
			raise ValueError("ai_generated score not found in API response")

		is_ai = score >= float(threshold)
		return is_ai, score, data

	except requests.RequestException:
		raise

	finally:
		# cleanup temp file if one was created by prepare_image_for_upload
		try:
			if 'is_temp' in locals() and is_temp and 'upload_path' in locals():
				os.remove(upload_path)
		except Exception:
			pass


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Check whether an image is AI-generated using Sightengine.")
	parser.add_argument("image", help="Image URL or local file path")
	parser.add_argument("--api_user", required=True, help="Sightengine api_user")
	parser.add_argument("--api_secret", required=True, help="Sightengine api_secret")
	parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold to consider image AI-generated")
	args = parser.parse_args()

	try:
		is_ai, score, raw = detect_ai_generated(args.image, args.api_user, args.api_secret, threshold=args.threshold)
		print(f"ai_generated_score={score:.4f}")
		print("AI-generated:" if is_ai else "Likely real:")
	except Exception as e:
		print("Error:", e)
