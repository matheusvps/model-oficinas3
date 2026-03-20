import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera dataset aumentado em disco a partir de dataset/train."
    )
    parser.add_argument("--input-dir", type=str, default="dataset", help="Dataset original com train/val/test.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_aug",
        help="Pasta de saida para dataset aumentado.",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=2,
        help="Quantas copias aumentadas gerar por imagem de treino.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Redimensionamento para augmentacao.")
    return parser.parse_args()


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(src_root: Path, dst_root: Path, split: str):
    src = src_root / split
    dst = dst_root / split
    if not src.exists():
        raise FileNotFoundError(f"Split nao encontrado: {src}")
    shutil.copytree(src, dst)


def build_augmenter():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        ],
        name="disk_augmenter",
    )


def save_image_uint8(image_tensor: tf.Tensor, out_path: Path):
    img = tf.clip_by_value(image_tensor, 0.0, 255.0)
    img = tf.cast(img, tf.uint8).numpy()
    encoded = tf.io.encode_jpeg(img, quality=95)
    tf.io.write_file(str(out_path), encoded)


def load_image_rgb(path: Path, img_size: int) -> tf.Tensor:
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def augment_train_split(
    src_train_dir: Path, dst_train_dir: Path, copies_per_image: int, img_size: int
):
    augmenter = build_augmenter()

    class_dirs = sorted([d for d in src_train_dir.iterdir() if d.is_dir()])
    total_generated = 0

    for class_dir in class_dirs:
        src_class = class_dir
        dst_class = dst_train_dir / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)

        image_files = sorted([p for p in src_class.iterdir() if p.is_file()])
        print(f"Classe: {class_dir.name} | originais: {len(image_files)}")

        for img_path in image_files:
            arr = load_image_rgb(img_path, img_size)

            stem = img_path.stem
            for i in range(copies_per_image):
                x = tf.expand_dims(arr, axis=0)
                aug = augmenter(x, training=True)[0]
                out_name = f"{stem}_aug{i+1}.jpg"
                out_path = dst_class / out_name
                save_image_uint8(aug, out_path)
                total_generated += 1

    return total_generated


def main():
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image deve ser >= 1")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_src = input_dir / "train"
    val_src = input_dir / "val"
    test_src = input_dir / "test"

    for d in [train_src, val_src, test_src]:
        if not d.exists():
            raise FileNotFoundError(f"Pasta nao encontrada: {d}")

    ensure_clean_dir(output_dir)

    print("Copiando val/test sem alteracoes...")
    copy_split(input_dir, output_dir, "val")
    copy_split(input_dir, output_dir, "test")

    print("Copiando train original para pasta de saida...")
    shutil.copytree(train_src, output_dir / "train")

    print("Gerando imagens aumentadas em train...")
    generated = augment_train_split(
        src_train_dir=train_src,
        dst_train_dir=output_dir / "train",
        copies_per_image=args.copies_per_image,
        img_size=args.img_size,
    )

    print(f"\nConcluido. Imagens aumentadas geradas: {generated}")
    print(f"Dataset aumentado salvo em: {output_dir}")
    print("Use este dataset no treino: --data-dir dataset_aug")


if __name__ == "__main__":
    main()
