#!/usr/bin/env python3
import argparse
import colorsys
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import axengine as ort
    BACKEND = "axengine"
except ImportError:
    import onnxruntime as ort
    BACKEND = "onnxruntime"

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

INPUT_SIZE = 640

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]


def load_font():
    for p in ("DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, 15)
        except OSError:
            continue
    return ImageFont.load_default()


def class_color(cls):
    r, g, b = colorsys.hsv_to_rgb((int(cls) * 0.618) % 1.0, 0.85, 0.92)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_box(draw, x1, y1, x2, y2, text, color, font):
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ty = y1 - th - 6 if y1 - th - 6 >= 0 else y1
    draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 6], fill=color)
    draw.text((x1 + 3, ty + 3), text, fill=(255, 255, 255), font=font)


def preprocess(img, size_dtype):
    w, h = img.size
    x = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = np.asarray(x, dtype=np.float32)[:, :, :3].transpose(2, 0, 1)[None]
    
    if BACKEND != "axengine":
        x = (x - MEAN) / STD
    return np.ascontiguousarray(x), np.array([[w, h]], dtype=size_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-i", "--image")
    ap.add_argument("-d", "--dir")
    ap.add_argument("-o", "--out", default="vis_axengine")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("-n", type=int, default=0)
    args = ap.parse_args()

    print(f"backend: {BACKEND}")
    sess = ort.InferenceSession(args.model)
    inputs = sess.get_inputs()
    print("inputs:", [(i.name, i.shape, getattr(i, "dtype", getattr(i, "type", "?"))) for i in inputs])

    meta = {i.name: i for i in inputs}
    size_meta = meta["orig_target_sizes"]
    size_dtype_str = str(getattr(size_meta, "dtype", getattr(size_meta, "type", "int32")))
    size_dtype = np.int64 if "int64" in size_dtype_str else np.int32

    if args.image:
        paths = [args.image]
    else:
        paths = [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
                 if f.lower().endswith((".jpg", ".png"))]
        if args.n > 0:
            paths = paths[: args.n]
    os.makedirs(args.out, exist_ok=True)

    import time
    font = load_font()
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        x, sizes = preprocess(img, size_dtype)
        t0 = time.time()
        labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": sizes})
        dt = (time.time() - t0) * 1000

        labels = labels.reshape(-1).astype(np.int32)
        scores = scores.reshape(-1)
        boxes = boxes.reshape(-1, 4)  

        draw = ImageDraw.Draw(img)
        ndet = 0
        for cls, score, (x1, y1, x2, y2) in zip(labels, scores, boxes):
            if score < args.thr:
                continue
            ndet += 1
            x1, x2 = sorted((max(0.0, min(float(x1), w)), max(0.0, min(float(x2), w))))
            y1, y2 = sorted((max(0.0, min(float(y1), h)), max(0.0, min(float(y2), h))))
            color = class_color(cls)
            name = COCO80[cls] if 0 <= cls < 80 else str(cls)
            draw_box(draw, int(x1), int(y1), int(x2), int(y2), f"{name} {score:.2f}", color, font)

        save = os.path.join(args.out, os.path.basename(p))
        img.save(save)
        print(f"{os.path.basename(p)}  dets={ndet}  infer={dt:.1f}ms  -> {save}")


if __name__ == "__main__":
    main()
