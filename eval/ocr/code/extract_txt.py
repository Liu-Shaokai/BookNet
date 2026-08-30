import argparse
import json
import os

EXCLUDE_LABELS = {"image", "chart", "header_image", "footer_image"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    total = 0
    folders = sorted(d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)))
    for folder in folders:
        src_dir = os.path.join(args.raw, folder)
        out_dir = os.path.join(args.out, folder)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json"))
        for fname in files:
            with open(os.path.join(src_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            parts = []
            for page in data.get("pages", []):
                pr = page.get("prunedResult", {})
                for block in pr.get("parsing_res_list", []):
                    if block.get("block_label") in EXCLUDE_LABELS:
                        continue
                    content = block.get("block_content")
                    if content is None:
                        continue
                    parts.append(str(content))
            text = "\n".join(parts)
            out_name = os.path.splitext(fname)[0] + ".txt"
            with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
                f.write(text)
            total += 1
        print(f"{folder}: {len(files)} files")
    print(f"Total txt files written: {total}")


if __name__ == "__main__":
    main()
