"""
entrenar_modelo.py
==================

Script de entrenamiento extraído y adaptado de
"Image classification with CNN modelo vision.ipynb" para poder regenerar el
modelo de clasificación de tizón tardío en hojas de papa (healthy /
late_blight) de forma independiente, fuera del notebook.

Pasos (idénticos a los del notebook original):
    1. Reorganiza el dataset ya preparado por preparacion_dataset/dataset_tizon_papa.py
       (data/potato_late_blight_dataset_validated/yolo/images/{train,val}) al formato
       ImageFolder esperado por PyTorch (dataset/{train,val}/<clase>).
    2. Carga ResNet-18 preentrenado, congela el extractor de características y
       reemplaza la capa de salida por una de 2 clases.
    3. Entrena con SGD + CyclicLR durante n_epochs, conservando los pesos de la
       época con mejor accuracy de validación.
    4. Guarda el modelo (state_dict) y las métricas por época.

Uso:
    python entrenamiento/entrenar_modelo.py
    python entrenamiento/entrenar_modelo.py --epocas 2 --salida-modelo /tmp/modelo_prueba.pt
"""

import argparse
import copy
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGENES_FUENTE = PROJECT_ROOT / "data" / "potato_late_blight_dataset_validated" / "yolo" / "images"
DATASET_IMAGEFOLDER = PROJECT_ROOT / "dataset"
MODELO_DEFECTO = PROJECT_ROOT / "models" / "model.pt"
METRICAS_DEFECTO = PROJECT_ROOT / "models" / "training_metrics_tizon_papa.csv"
CLASS_NAMES = ["healthy", "late_blight"]

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
])


def seleccionar_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_class(filename: str) -> str:
    return "late_blight" if filename.startswith("late_blight_") else "healthy"


def preparar_imagefolder() -> None:
    """Reorganiza train/val del dataset validado a la estructura dataset/<split>/<clase>."""
    if not IMAGENES_FUENTE.is_dir():
        raise FileNotFoundError(
            f"No se encontró {IMAGENES_FUENTE}. Ejecuta primero "
            "preparacion_dataset/dataset_tizon_papa.py."
        )

    if DATASET_IMAGEFOLDER.exists():
        shutil.rmtree(DATASET_IMAGEFOLDER)

    for split in ("train", "val"):
        split_dir = IMAGENES_FUENTE / split
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = infer_class(img_path.name)
            out_dir = DATASET_IMAGEFOLDER / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_dir / img_path.name)

    print(f"Dataset ImageFolder preparado en: {DATASET_IMAGEFOLDER}")


def train_model(model, train_loader, validation_loader, val_dataset, criterion,
                 optimizer, scheduler, n_epochs, device):
    loss_list = []
    accuracy_list = []
    n_val = len(val_dataset)
    accuracy_best = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        loss_sublist = []
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = criterion(z, y)
            loss_sublist.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        loss_list.append(float(np.mean(loss_sublist)))

        correct = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                z = model(x_val)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_val).sum().item()

        accuracy = correct / n_val
        accuracy_list.append(accuracy)

        if accuracy > accuracy_best:
            accuracy_best = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Época {epoch + 1}/{n_epochs} | lr={optimizer.param_groups[0]['lr']:.5f} "
              f"| pérdida_train={loss_list[-1]:.4f} | accuracy_val={accuracy:.4f}")

    model.load_state_dict(best_model_wts)
    return accuracy_list, loss_list, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de tizón tardío en papa")
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--salida-modelo", type=str, default=str(MODELO_DEFECTO))
    parser.add_argument("--salida-metricas", type=str, default=str(METRICAS_DEFECTO))
    args = parser.parse_args()

    try:
        preparar_imagefolder()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    device = seleccionar_dispositivo()
    print(f"Dispositivo de entrenamiento: {device}")

    train_dataset = ImageFolder(str(DATASET_IMAGEFOLDER / "train"), transform=TRANSFORM)
    val_dataset = ImageFolder(str(DATASET_IMAGEFOLDER / "val"), transform=TRANSFORM)

    if train_dataset.classes != CLASS_NAMES:
        print(f"Aviso: el orden de clases detectado es {train_dataset.classes}, "
              f"se esperaba {CLASS_NAMES}.", file=sys.stderr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2"
    )

    inicio = time.time()
    print(f"Inicio de entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    accuracy_list, loss_list, model = train_model(
        model, train_loader, validation_loader, val_dataset,
        criterion, optimizer, scheduler, args.epocas, device,
    )
    print(f"Entrenamiento completado en {time.time() - inicio:.2f} s")

    ruta_modelo = Path(args.salida_modelo)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    ruta_metricas = Path(args.salida_metricas)
    ruta_metricas.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame({
        "epoch": range(1, args.epocas + 1),
        "train_loss": loss_list,
        "val_accuracy": accuracy_list,
    })
    metrics_df.to_csv(ruta_metricas, index=False)
    print(f"Métricas guardadas en: {ruta_metricas}")
    print(f"Mejor accuracy de validación: {max(accuracy_list):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
