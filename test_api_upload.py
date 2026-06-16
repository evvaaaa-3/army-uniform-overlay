import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Upload a local image to the Flask photo try-on API.")
    parser.add_argument("image_path", help="Path to the person photo to upload.")
    parser.add_argument("--mode", choices=["upper_body", "full_body"], default="upper_body")
    parser.add_argument("--debug", action="store_true", help="Ask the API to save and return a debug image.")
    parser.add_argument("--url", default="http://127.0.0.1:5050/process-photo")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    try:
        with image_path.open("rb") as image_file:
            response = requests.post(
                args.url,
                files={"image": (image_path.name, image_file)},
                data={"mode": args.mode, "debug": "true" if args.debug else "false"},
                timeout=120,
            )
    except requests.RequestException as exc:
        print(f"Request failed before a response was received.")
        print(f"Target URL: {args.url}")
        print(f"Error: {exc}")
        raise SystemExit(1)

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_response": response.text}

    print(json.dumps(payload, indent=2))

    if not response.ok:
        print(f"Status code: {response.status_code}")
        print(f"Target URL: {args.url}")
        print(f"Raw response: {response.text}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
