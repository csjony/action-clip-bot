from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import subprocess

app = FastAPI()

# Generate dummy video file
if not os.path.exists("dummy.mp4"):
    print("Generating dummy MP4 video...")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=16", 
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "dummy.mp4"],
        check=True
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"status": "ready"}

@app.post("/generate")
def generate(payload: dict):
    print("Mock queueing video for prompt:", payload.get("prompt"))
    return {"job_id": "mock_job_123"}

@app.get("/status/{job_id}")
def status(job_id: str):
    return {"status": "done"}

@app.get("/result/{job_id}")
def result(job_id: str):
    if not os.path.exists("dummy.mp4"):
        raise HTTPException(status_code=500, detail="Dummy video not found")
    return FileResponse("dummy.mp4", media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
