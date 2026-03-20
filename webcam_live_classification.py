import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mostra webcam ao vivo + classificacao da laranja em tempo real."
    )
    parser.add_argument("--model", type=str, default="artifacts/orange_model.keras")
    parser.add_argument("--labels", type=str, default="artifacts/labels.json")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam USB.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confianca minima.")
    parser.add_argument("--box-size", type=float, default=0.55, help="Tamanho do quadro central (0.3 a 0.9).")
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
    return np.expand_dims(x, axis=0)


def center_box(frame, ratio):
    h, w = frame.shape[:2]
    ratio = max(0.30, min(0.90, ratio))
    box_w = int(w * ratio)
    box_h = int(h * ratio)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    return x1, y1, x2, y2


def draw_header(frame, line1, line2, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 84), (0, 0, 0), -1)
    cv2.putText(frame, line1, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels nao encontrados: {labels_path}")

    model, class_names, img_size, binary_map = load_assets(model_path, labels_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir webcam no indice {args.camera}.")

    print("Janela aberta. Pressione Q para sair.")
    print("Dica: centralize a laranja dentro do retangulo verde.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = center_box(frame, args.box_size)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        x = preprocess(roi, img_size)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        class_name = class_names[pred_idx]
        health_status = binary_map[pred_idx]

        if conf < args.threshold:
            status = "INCONCLUSIVO"
            status_color = (0, 255, 255)
        elif health_status == "healthy":
            status = "SAUDAVEL"
            status_color = (0, 220, 0)
        else:
            status = "DOENTE"
            status_color = (0, 0, 255)

        draw_header(
            frame,
            f"Classificacao: {status} | Classe: {class_name} | Conf: {conf:.2f}",
            "Q: sair | Webcam ao vivo com classificacao em tempo real",
            status_color,
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            "Area analisada",
            (x1, max(96, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        roi_preview = cv2.resize(roi, (220, 220), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        py1, py2 = h - 230, h - 10
        px1, px2 = w - 230, w - 10
        frame[py1:py2, px1:px2] = roi_preview
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 1)
        cv2.putText(
            frame,
            "Recorte usado",
            (px1, py1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Laranja - Webcam + Classificacao", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
