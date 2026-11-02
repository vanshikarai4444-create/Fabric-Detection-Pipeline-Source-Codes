"""Run the Roboflow-hosted fabric classifier on 50 unseen test images and export predictions.csv.

Usage:
    export ROBOFLOW_API_KEY=...        # never hard-code the key
    python infer.py [DATA_DIR] [N_IMAGES]

DATA_DIR is the Roboflow "folder" export of dataset v2 (contains train/ valid/ test/).
If test_50.txt exists (written by the CNN notebook), the same images are used so both
models are scored on identical inputs; otherwise N images are sampled from test/ with seed 42.
"""
import csv
import os
import random
import sys
import time

from inference_sdk import InferenceHTTPClient

MODEL_ID = "fabric-detection-pipeline/2"
API_URL = "https://serverless.roboflow.com"
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/data"
N_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUT_CSV = "predictions.csv"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pick_test_images():
    """Return paths relative to DATA_DIR, e.g. 'test/defective/abc.jpg'."""
    if os.path.exists("test_50.txt"):
        return [p.strip() for p in open("test_50.txt") if p.strip()][:N_IMAGES]
    test_dir = os.path.join(DATA_DIR, "test")
    images = []
    # Same order as torchvision ImageFolder (sorted classes, sorted files), so seed 42 picks the same images
    for cls in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir) or cls.lower() == "unlabeled":
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith(IMG_EXTS):
                images.append(os.path.join("test", cls, name))
    idx = random.Random(42).sample(range(len(images)), N_IMAGES)
    return [images[i] for i in idx]


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set the ROBOFLOW_API_KEY environment variable first.")
    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)

    rows, latencies = [], []
    rel_paths = pick_test_images()
    for i, rel in enumerate(rel_paths, 1):
        path = os.path.join(DATA_DIR, rel)
        actual = os.path.basename(os.path.dirname(path))  # class = parent folder name

        start = time.perf_counter()
        result = client.infer(path, model_id=MODEL_ID)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Single-label classification returns "top" + "confidence"; fall back to the predictions list
        predicted = result.get("top") or result["predictions"][0]["class"]
        confidence = result.get("confidence") or result["predictions"][0]["confidence"]

        rows.append({
            "image_name": os.path.basename(path),
            "actual_label": actual,
            "predicted_label": predicted,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(latency_ms, 1),
            "correct": predicted == actual,
        })
        print(f"[{i:2d}/{len(rel_paths)}] {os.path.basename(path)}: {predicted} ({confidence:.3f}) {latency_ms:.0f} ms")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nSaved {len(rows)} predictions to {OUT_CSV}")
    print(f"Accuracy on {len(rows)} images: {accuracy:.2%}")
    print(f"Average latency: {sum(latencies) / len(latencies):.1f} ms/image (includes network round trip)")


if __name__ == "__main__":
    main()
