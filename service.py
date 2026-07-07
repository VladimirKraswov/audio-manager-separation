#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the audio manager separation API service.")
    parser.add_argument("--host", default=os.environ.get("AMS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AMS_PORT", "8088")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("src.service_app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
