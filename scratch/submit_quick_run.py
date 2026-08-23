import httpx, json

script_data = {
  "title": "Neon Chase Quick Test",
  "theme": "neon_city_chase",
  "hook": "A neon speed car chase through cyber streets.",
  "narration": "They thought they could lock down the city grid. They were wrong. Hit the overdrive.",
  "hashtags": ["#action", "#cyberpunk", "#shorts"],
  "captions": {
    "youtube": "Neon city sports car chase action scene. #action #cyberpunk #shorts",
    "facebook": "Neon city sports car chase action scene. #action #cyberpunk #reels",
    "instagram": "Neon city sports car chase action scene. #action #cyberpunk #reels",
    "threads": "Neon city sports car chase action scene. #action #cyberpunk #reels",
    "tiktok": "Neon city sports car chase action scene. #action #cyberpunk #fyp"
  },
  "scenes": [
    {"index": 0, "prompt": "Futuristic cyan and magenta neon glowing sports car speeding down wet city street, high angle shot, hyperdetailed cinematic render", "duration_sec": 5, "sound_query": "sports car engine acceleration"},
    {"index": 1, "prompt": "Cyber sports car drifting hard around a sharp corner, spray particles flying, wet asphalt reflecting neon signs, cinematic action shot", "duration_sec": 5, "sound_query": "tire drift screech"},
    {"index": 2, "prompt": "Close-up of futuristic glowing speedometer counting up rapidly, digital interface, motion blur background, dramatic lighting", "duration_sec": 5, "sound_query": "sci-fi telemetry beep"}
  ]
}

resp = httpx.post(
    "http://localhost:8080/generate",
    data={"theme": "neon_city_chase", "quick_test": "true", "script_json": json.dumps(script_data)},
    follow_redirects=False,
)
print("Status:", resp.status_code)
if resp.status_code in (302, 303):
    run_id = resp.headers.get("location", "").split("/")[-1]
    print("Run ID:", run_id)
    print(f"Monitor at: http://localhost:8080/runs/{run_id}")
    print(f"Live logs:  http://localhost:8080/logs")
else:
    print("Response:", resp.text[:500])
