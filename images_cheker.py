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
import random
from typing import List, Dict

try:
	import cv2
	import numpy as np
except Exception:
	cv2 = None
	np = None

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


def check_video_frames(video_path: str, api_user: str, api_secret: str, *, sample_count: int = 5, quality: int = 75, max_bytes: int = 12 * 1024 * 1024, timeout: int = 15) -> List[Dict]:
	"""Sample random frames from a video, save as WebP, run AI-detection on each.

	Returns a list of dicts: {"frame_index": int, "is_ai": bool, "score": float, "raw": dict}

	Requires `opencv-python` and `numpy`. If OpenCV is not available an error is raised.
	Temporary WebP files are removed before returning.
	"""
	if cv2 is None or np is None:
		raise RuntimeError("opencv-python and numpy are required to sample video frames")

	if not os.path.isfile(video_path):
		raise ValueError(f"Video file not found: {video_path}")

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError("Unable to open video file")

	try:
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		if frame_count <= 0:
			# fallback: iterate to count frames (slow)
			frame_count = 0
			while True:
				ret, _ = cap.read()
				if not ret:
					break
				frame_count += 1
			cap.release()
			cap = cv2.VideoCapture(video_path)

		# choose sample indices
		sample_count = max(1, int(sample_count))
		if frame_count <= sample_count:
			indices = list(range(frame_count))
		else:
			indices = random.sample(range(frame_count), sample_count)

		results: List[Dict] = []
		temp_files: List[str] = []

		for idx in indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ret, frame = cap.read()
			if not ret or frame is None:
				continue

			# convert BGR -> RGB
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# convert to PIL image for fast webp save
			try:
				from PIL import Image as PilImage
				im = PilImage.fromarray(rgb)
			except Exception:
				# fallback: write using OpenCV (will be png/jpg) then convert
				tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
				cv2.imwrite(tf.name, frame)
				try:
					from PIL import Image as PilImage
					im = PilImage.open(tf.name).convert("RGB")
				finally:
					try:
						os.remove(tf.name)
					except Exception:
						pass

			# write to temp webp file
			tf = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
			temp_files.append(tf.name)
			try:
				im.save(tf.name, format="WEBP", quality=int(quality), method=6)
			except Exception:
				im.save(tf.name, format="WEBP")

			# ensure size constraint; if too large, run prepare_image_for_upload
			try:
				if os.path.getsize(tf.name) > max_bytes:
					small_path, is_temp = prepare_image_for_upload(tf.name, max_bytes=max_bytes)
					if is_temp and small_path != tf.name:
						# replace temp file
						try:
							os.remove(tf.name)
						except Exception:
							pass
						tf.name = small_path
						temp_files[-1] = small_path
			except Exception:
				pass

			# run detection
			try:
				is_ai, score, raw = detect_ai_generated(tf.name, api_user, api_secret, timeout=timeout)
			except Exception as e:
				is_ai, score, raw = False, 0.0, {"error": str(e)}

			results.append({"frame_index": int(idx), "is_ai": bool(is_ai), "score": float(score), "raw": raw})

		return results

	finally:
		try:
			cap.release()
		except Exception:
			pass
		# cleanup temp files
		for p in locals().get("temp_files", []):
			try:
				if p and os.path.exists(p):
					os.remove(p)
			except Exception:
				pass


def summarize_frame_results(results: List[Dict], threshold: float = 0.5, percent_threshold: float = 0.3, max_threshold: float = 0.9, avg_threshold: float = 0.6) -> Dict:
	"""Produce a concise summary and decision from frame-level results.

	Decision rules (defaults):
	  - AI if >= `percent_threshold` of frames have score >= `threshold`
	  - OR if max score >= `max_threshold`
	  - OR if average score >= `avg_threshold`
	"""
	scores = [float(r.get("score", 0.0)) for r in results]
	n = len(scores)
	if n == 0:
		return {"n_frames": 0, "avg_score": 0.0, "max_score": 0.0, "frames_above": 0, "percent_above": 0.0, "decision": False, "reason": "no_frames"}
	avg = sum(scores) / n
	mx = max(scores)
	frames_above = sum(1 for s in scores if s >= threshold)
	percent_above = frames_above / n
	decision = (percent_above >= percent_threshold) or (mx >= max_threshold) or (avg >= avg_threshold)

	if mx >= max_threshold:
		reason = f"max_score {mx:.2f} >= {max_threshold}"
	elif percent_above >= percent_threshold:
		reason = f"{percent_above:.2%} frames >= {threshold}"
	elif avg >= avg_threshold:
		reason = f"avg_score {avg:.2f} >= {avg_threshold}"
	else:
		reason = "no strong AI signal"

	return {
		"n_frames": n,
		"avg_score": round(avg, 4),
		"max_score": round(mx, 4),
		"frames_above": frames_above,
		"percent_above": round(percent_above, 4),
		"decision": bool(decision),
		"reason": reason,
	}


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
