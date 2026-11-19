import csv
import os
import time
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from pathlib import Path

import requests

CSV_FILE = "hamos_image_urls.csv"
OUTPUT_DIR = Path("images")
REPORT_FILE = "download_report.csv"

# Keep this modest to avoid putting unnecessary load on the image host.
MAX_WORKERS = 4
MAX_RETRIES = 3
TIMEOUT = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_extension(data, content_type, url):
    # Prefer file signatures over the URL suffix or server header.
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:3] == b"GIF":
        return ".gif"

    content_type = (content_type or "").split(";")[0].strip().lower()
    guessed = mimetypes.guess_extension(content_type)
    if guessed in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if guessed == ".jpeg" else guessed

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    return None


def download_one(row):
    image_id = row["image_id"].strip()
    url = row["photo_url"].strip()

    if not image_id or not url:
        return {"image_id": image_id, "status": "skipped", "path": "", "error": "Missing ID or URL"}

    # Skip an image if any supported extension already exists.
    for ext in (".jpg", ".png", ".webp", ".gif"):
        existing = OUTPUT_DIR / f"{image_id}{ext}"
        if existing.exists() and existing.stat().st_size > 0:
            return {"image_id": image_id, "status": "already_exists",
                    "path": str(existing), "error": ""}

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=TIMEOUT,
                headers={"User-Agent": "HAMOS-dataset-research/1.0"}
            )

            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", "5"))
                time.sleep(min(wait, 60))
                last_error = "HTTP 429 rate limited"
                continue

            response.raise_for_status()
            data = response.content

            if not data:
                raise ValueError("Empty response")

            ext = detect_extension(data, response.headers.get("Content-Type"), url)
            if ext is None:
                raise ValueError(
                    f"Response does not look like a supported image "
                    f"(Content-Type: {response.headers.get('Content-Type')})"
                )

            destination = OUTPUT_DIR / f"{image_id}{ext}"
            temp_path = destination.with_suffix(destination.suffix + ".part")
            temp_path.write_bytes(data)
            os.replace(temp_path, destination)

            return {"image_id": image_id, "status": "downloaded",
                    "path": str(destination), "error": ""}

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    return {"image_id": image_id, "status": "failed", "path": "", "error": last_error}


def main():
    if not Path(CSV_FILE).exists():
        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. Put it in the same folder as this script."
        )

    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} image records.")
    print(f"Downloading with {MAX_WORKERS} workers into: {OUTPUT_DIR.resolve()}")
    print("Only download images where access is permitted by the source platform.")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, row) for row in rows]

        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 100 == 0 or index == len(futures):
                print(f"Processed {index}/{len(futures)}")

    # Preserve input order in the report.
    order = {row["image_id"]: i for i, row in enumerate(rows)}
    results.sort(key=lambda x: order.get(x["image_id"], 10**12))

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "status", "path", "error"]
        )
        writer.writeheader()
        writer.writerows(results)

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    print("\nFinished.")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Report saved to: {Path(REPORT_FILE).resolve()}")


if __name__ == "__main__":
    main()
