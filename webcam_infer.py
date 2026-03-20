import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferência em tempo real via webcam USB para doença de laranja."
    )
    parser.add_argument("--model", type=str, default="artifacts/orange_model.keras")
    parser.add_argument("--labels", type=str, default="artifacts/labels.json")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam USB.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confiança mínima.")
    return parser.parse_args()


def load_assets(model_path: Path, labels_path: Path):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata["class_names"]
    img_size = int(metadata["img_size"])
    binary_map = {int(k): v for k, v in metadata["binary_map"].items()}
    return model, class_names, img_size, binary_map


def preprocess(frame_bgr, img_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x


def overlay_text(frame, text, color=(0, 255, 0)):
    cv2.rectangle(frame, (10, 10), (890, 90), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels não encontrados: {labels_path}")

    model, class_names, img_size, binary_map = load_assets(model_path, labels_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir webcam no índice {args.camera}.")

    print("Webcam iniciada. Pressione 'q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        x = preprocess(frame, img_size)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        disease_class = class_names[pred_idx]
        health_status = binary_map[pred_idx]

        if conf < args.threshold:
            label = f"Inconclusivo ({conf:.2f})"
            color = (0, 255, 255)
        else:
            if health_status == "healthy":
                color = (0, 220, 0)
                label = f"SAUDAVEL | classe={disease_class} | conf={conf:.2f}"
            else:
                color = (0, 0, 255)
                label = f"DOENTE | classe={disease_class} | conf={conf:.2f}"

        overlay_text(frame, label, color=color)
        cv2.imshow("Orange Disease Detection (USB Webcam)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
