from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle recognition with YOLOv8")
    parser.add_argument("--source", type=str, default="0", help="Image, video path, or camera index")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--show", action="store_true", help="Display result window")
    parser.add_argument("--save", action="store_true", help="Save output to runs/vehicle_recognition")
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def draw_detections(frame, result, names: dict[int, str]):
    counts: Counter[str] = Counter()
    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        if label not in VEHICLE_CLASSES:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        counts[label] += 1

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    summary = " | ".join([f"{k}: {v}" for k, v in sorted(counts.items())]) or "No vehicles"
    cv2.putText(
        frame,
        f"Vehicles -> {summary}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    source = normalize_source(args.source)

    model = YOLO(args.model)

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        stream=True,
        verbose=False,
    )

    out_dir = Path("runs/vehicle_recognition")
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    window_name = "Vehicle Recognition"

    for result in results:
        frame = result.orig_img.copy()
        frame = draw_detections(frame, result, model.names)

        if args.show:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                output_path = out_dir / "output.mp4"
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25,
                    (w, h),
                )
            writer.write(frame)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
