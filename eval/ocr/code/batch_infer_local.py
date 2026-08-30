import argparse
import base64
import json
import os
import time

import requests

IMG_EXTS = (".jpg", ".jpeg", ".png")
MAX_RETRIES = 3
TIMEOUT = 300


def already_done(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("pages", [{}])[0].get("markdown"))
    except Exception:
        return False


def process_one(endpoint, src_path, out_path):
    with open(src_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("ascii")
    payload = {"file": image_data, "fileType": 1}

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(endpoint + "/layout-parsing", json=payload, timeout=TIMEOUT)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(2 * attempt)
                continue
            result = resp.json()["result"]
            output = {
                "image": os.path.basename(src_path),
                "pages": [
                    {
                        "prunedResult": r["prunedResult"],
                        "markdown": r.get("markdown", {}).get("text"),
                    }
                    for r in result["layoutParsingResults"]
                ],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            return True, None
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * attempt)
    return False, last_err


def main():
    ap = argparse.ArgumentParser(description="Run layout-parsing OCR on every image in a directory.")
    ap.add_argument("--src", required=True, help="directory containing input images")
    ap.add_argument("--out", required=True, help="directory to write per-image JSON results")
    ap.add_argument("--endpoint", default="http://localhost:8181")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(IMG_EXTS))
    total = len(files)
    print(f"Total images: {total}")

    done = failed = skipped = 0
    failures = []

    for i, fname in enumerate(files, 1):
        src_path = os.path.join(args.src, fname)
        out_path = os.path.join(args.out, os.path.splitext(fname)[0] + ".json")

        if already_done(out_path):
            skipped += 1
        else:
            ok, err = process_one(args.endpoint, src_path, out_path)
            if ok:
                done += 1
            else:
                failed += 1
                failures.append((fname, err))

        if i % 20 == 0 or i == total:
            print(f"[{i}/{total}] done={done} failed={failed} skipped={skipped}", flush=True)

    print(f"\n=== DONE === done={done} failed={failed} skipped={skipped} total={total}")
    if failures:
        for fname, err in failures[:20]:
            print(f"  {fname}: {err}")
        with open(os.path.join(args.out, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()