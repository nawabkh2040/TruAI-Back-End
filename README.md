# TruAI Back End

Small service to check whether images or sampled video frames are AI-generated using the Sightengine `genai` model.

Prerequisites
- Python 3.8+
- `pip` for installing dependencies

Install
```powershell
pip install -r requirements.txt
```

Environment
- `SIGHTENGINE_API_USER` — your Sightengine api_user
- `SIGHTENGINE_API_SECRET` — your Sightengine api_secret

Quick start (development)
```powershell
# $env:SIGHTENGINE_API_USER = '123'
# $env:SIGHTENGINE_API_SECRET = 'secret'
python -m uvicorn app:app --reload --port 8000
```

Endpoints
- `POST /check` — accepts a single image file; returns `{is_ai, score, raw}`
  - Form fields: `file` (file upload), optional `threshold` (float)

- `POST /check_video` — accepts a video file; samples frames across the video, runs detection, and returns `summary` and `frames` details
  - Form fields: `file` (video upload), optional `sample_count` (int, 0 = auto), optional `threshold` (float)

Examples
Image:
```bash
curl -F "file=@/path/to/image.jpg" localhost:8000/check
```
Video:
```bash
curl -F "file=@/path/to/video.mp4" -F "sample_count=0" localhost:8000/check_video
```

Notes
- The server reads Sightengine credentials from environment variables. Do not send credentials in requests.
- For video sampling the service depends on `opencv-python`, `numpy` and `pillow` to save frames as WebP. Install them if you plan to use `/check_video`.
- You can tweak thresholds in `images_cheker.summarize_frame_results()`.

Files
- `images_cheker.py` — detection and helper functions
- `app.py` — FastAPI server with `/check` and `/check_video`
- `requirements.txt` — Python dependencies


