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
    # Compatibilidade com comandos antigos: argumento aceito, mas ignorado.
    parser.add_argument("--camera", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--threshold", type=float, default=0.60, help="Confianca minima.")
    parser.add_argument(
        "--inconclusive-margin",
        type=float,
        default=0.08,
        help="Margem abaixo do limiar para marcar INCONCLUSIVO (histerese).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="Peso da predicao atual na suavizacao temporal (0 a 1).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.06,
        help="Diferenca minima entre top-1 e top-2 para confirmar classe.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=6,
        help="Frames seguidos para considerar previsao estavel.",
    )
    parser.add_argument(
        "--min-fruit-area-ratio",
        type=float,
        default=0.08,
        help="Area minima da fruta detectada dentro do quadro fixo (0 a 1).",
    )
    parser.add_argument(
        "--healthy-bias",
        type=float,
        default=0.05,
        help="Vantagem para decidir SAUDAVEL quando healthy estiver proximo da classe top.",
    )
    parser.add_argument(
        "--min-healthy-conf",
        type=float,
        default=0.35,
        help="Confianca minima da classe healthy para permitir override de SAUDAVEL.",
    )
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


def detect_fruit_box(roi_bgr, min_area_ratio):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Faixas para laranja verde/amarelada e casca mais saturada.
    mask_green = cv2.inRange(hsv, (28, 25, 25), (95, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 35, 35), (35, 255, 255))
    mask = cv2.bitwise_or(mask_green, mask_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = roi_bgr.shape[:2]
    min_area = max(1.0, float(min_area_ratio) * float(h * w))
    best = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        rect_area = max(1.0, float(bw * bh))
        fill_ratio = float(area) / rect_area
        score = area * fill_ratio
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)
    return best


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

    healthy_idx = next((i for i, name in enumerate(class_names) if name.lower() == "healthy"), None)

    camera_idx = 0
    print(f"Usando camera no indice {camera_idx}.")
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir webcam no indice {camera_idx}.")

    print("Janela aberta. Pressione Q para sair.")
    print("Dica: centralize a laranja dentro do retangulo verde.")

    smoothed_probs = None
    prev_pred_idx = None
    same_class_streak = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = center_box(frame, args.box_size)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        fruit_box = detect_fruit_box(roi, args.min_fruit_area_ratio)
        if fruit_box is None:
            status = "INCONCLUSIVO"
            class_name = "sem_fruta_detectada"
            conf = 0.0
            second_conf = 0.0
            top_margin = 0.0
            same_class_streak = 0
            prev_pred_idx = None
            status_color = (0, 255, 255)
            fruit_crop = roi
            detected_global = None
        else:
            fx, fy, fw, fh = fruit_box
            pad = int(0.08 * max(fw, fh))
            fx1 = max(0, fx - pad)
            fy1 = max(0, fy - pad)
            fx2 = min(roi.shape[1], fx + fw + pad)
            fy2 = min(roi.shape[0], fy + fh + pad)

            fruit_crop = roi[fy1:fy2, fx1:fx2]
            detected_global = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)

            x = preprocess(fruit_crop, img_size)
            probs = model.predict(x, verbose=0)[0]
            alpha = max(0.0, min(1.0, float(args.ema_alpha)))
            if smoothed_probs is None:
                smoothed_probs = probs
            else:
                smoothed_probs = (1.0 - alpha) * smoothed_probs + alpha * probs

            pred_idx = int(np.argmax(smoothed_probs))
            conf = float(smoothed_probs[pred_idx])
            sorted_probs = np.sort(smoothed_probs)
            second_conf = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
            top_margin = conf - second_conf
            class_name = class_names[pred_idx]
            health_status = binary_map[pred_idx]
            healthy_conf = float(smoothed_probs[healthy_idx]) if healthy_idx is not None else 0.0

            if pred_idx == prev_pred_idx:
                same_class_streak += 1
            else:
                same_class_streak = 1
                prev_pred_idx = pred_idx

            effective_threshold = max(0.0, min(1.0, args.threshold - args.inconclusive_margin))
            is_stable = same_class_streak >= max(1, int(args.stable_frames))
            stable_relaxed_threshold = max(0.0, effective_threshold - 0.07)
            low_confidence = conf < effective_threshold
            weak_margin = top_margin < max(0.0, float(args.min_margin))
            cannot_relax = (not is_stable) or (conf < stable_relaxed_threshold)
            healthy_close = healthy_conf >= (conf - max(0.0, float(args.healthy_bias)))
            healthy_override = healthy_close and (healthy_conf >= max(0.0, float(args.min_healthy_conf)))

            if healthy_override:
                status = "SAUDAVEL"
                class_name = "healthy"
                status_color = (0, 220, 0)
            elif low_confidence and weak_margin and cannot_relax:
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
            f"Top2: {second_conf:.2f} | Margem: {top_margin:.2f} | Estavel: {same_class_streak}f | Q: sair",
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
        if detected_global is not None:
            gx1, gy1, gx2, gy2 = detected_global
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 120, 0), 2)
            cv2.putText(
                frame,
                "Laranja detectada",
                (gx1, max(96, gy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 120, 0),
                2,
                cv2.LINE_AA,
            )

        roi_preview = cv2.resize(fruit_crop, (220, 220), interpolation=cv2.INTER_AREA)
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
