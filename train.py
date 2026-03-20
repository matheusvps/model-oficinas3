import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina classificador de doenças em laranja (multiclasse)."
    )
    parser.add_argument("--data-dir", type=str, default="dataset", help="Pasta com train/val/test.")
    parser.add_argument("--img-size", type=int, default=224, help="Tamanho da imagem quadrada.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Pasta de saída.")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Se ativo, faz fine-tuning de parte da base MobileNetV2 no final.",
    )
    return parser.parse_args()


def count_images_per_class(directory: Path, class_names):
    counts = {}
    total = 0
    for class_name in class_names:
        class_dir = directory / class_name
        n = sum(1 for p in class_dir.iterdir() if p.is_file())
        counts[class_name] = n
        total += n
    return counts, total


def build_model(num_classes: int, img_size: int):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="orange_disease_classifier")
    return model, base


def make_binary_map(class_names):
    mapping = {}
    for idx, name in enumerate(class_names):
        mapping[idx] = "healthy" if name.lower() == "healthy" else "diseased"
    return mapping


def evaluate_split(model, ds, split_name):
    loss, acc = model.evaluate(ds, verbose=0)
    print(f"[{split_name}] loss={loss:.4f} acc={acc:.4f}")
    return {"loss": float(loss), "acc": float(acc)}


def main():
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {d}")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    train_counts, train_total = count_images_per_class(train_dir, class_names)
    class_weight = {}
    for idx, class_name in enumerate(class_names):
        n = train_counts[class_name]
        class_weight[idx] = train_total / (num_classes * max(1, n))

    model, base_model = build_model(num_classes=num_classes, img_size=args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    print("\nTreino fase 1 (feature extractor congelado)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if args.fine_tune:
        print("\nTreino fase 2 (fine-tuning parcial)...")
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr * 0.1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(4, args.epochs // 2),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

    print("\nAvaliação final:")
    val_metrics = evaluate_split(model, val_ds, "val")
    test_metrics = evaluate_split(model, test_ds, "test")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "orange_model.keras"
    model.save(model_path)

    metadata = {
        "img_size": args.img_size,
        "class_names": class_names,
        "binary_map": make_binary_map(class_names),
        "train_counts": train_counts,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Metadados salvos em: {out_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
