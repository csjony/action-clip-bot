"""Entry point: python -m src.dashboard"""
import os
import sys

import uvicorn


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    print(f"Action Clip Bot Dashboard → http://{host}:{port}", file=sys.stderr)
    uvicorn.run("src.dashboard.app:app", host=host, port=port, reload=False,
                log_level="info")


if __name__ == "__main__":
    main()
