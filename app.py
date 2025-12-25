from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
from typing import Any

try:
    # local import of the detector implemented earlier
    from images_cheker import detect_ai_generated
except Exception as e:
    detect_ai_generated = None  # will error at runtime if missing


app = FastAPI(title="AI Image Checker")


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "ai-image-checker"}


@app.post("/check_image")
async def check_image(file: UploadFile = File(...), threshold: float = Form(0.5)) -> Any:
    """Receive an uploaded image, save locally, run AI-detection, delete file, return result.
    """
    if detect_ai_generated is None:
        raise HTTPException(status_code=500, detail="images_cheker.detect_ai_generated not available")

    api_user = os.environ.get("SIGHTENGINE_API_USER")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET")
    if not api_user or not api_secret:
        raise HTTPException(status_code=400, detail="Missing SIGHTENGINE_API_USER or SIGHTENGINE_API_SECRET in environment")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ""
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

        is_ai, score, raw = detect_ai_generated(tmp_path, api_user, api_secret, threshold=threshold)

        return JSONResponse({"is_ai": bool(is_ai), "score": float(score), "raw": raw})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.post("/check_video")
async def check_video(file: UploadFile = File(...), sample_count: int = Form(0), threshold: float = Form(0.5)) -> Any:
    """Accept a video upload, sample frames, summarize AI-generation likelihood, and return summary."""
    if detect_ai_generated is None:
        raise HTTPException(status_code=500, detail="images_cheker.detect_ai_generated not available")

    api_user = os.environ.get("SIGHTENGINE_API_USER")
    api_secret = os.environ.get("SIGHTENGINE_API_SECRET")
    if not api_user or not api_secret:
        raise HTTPException(status_code=400, detail="Missing SIGHTENGINE_API_USER or SIGHTENGINE_API_SECRET in environment")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ""
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

        # lazy import to avoid requiring cv2 for image-only use
        from images_cheker import check_video_frames, summarize_frame_results

        # if client passed 0 (default), let checker auto-select sample count
        try:
            frames = check_video_frames(tmp_path, api_user, api_secret, sample_count=(None if sample_count <= 0 else sample_count))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e) + ". Install opencv-python and numpy to enable video sampling: pip install opencv-python numpy")

        summary = summarize_frame_results(frames, threshold=threshold)

        return JSONResponse({"summary": summary, "frames": frames})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
