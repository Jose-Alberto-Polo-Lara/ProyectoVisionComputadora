#!/usr/bin/env python
# coding: utf-8

# # Dataset de detección de tizón tardío en papa
# 
# Este notebook audita el dataset anotado, corrige fugas entre particiones, exporta a YOLO, COCO y Pascal VOC, y genera los documentos de entrega.
# 
# > Las imágenes están balanceadas entre `healthy` y `late_blight`. Solo `late_blight` es una clase de objeto; las imágenes sanas se conservan como ejemplos negativos sin cajas.

# In[10]:


from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
SOURCE_DIR = Path.cwd() / "potato_late_blightanotaciones"
OUTPUT_DIR = Path.cwd() / "potato_late_blight_dataset_validated"
SPLITS = ("train", "val", "test")
IMAGE_CLASSES = ("healthy", "late_blight")

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"No se encontró el dataset anotado: {SOURCE_DIR}")

print(f"Dataset fuente: {SOURCE_DIR}")
print(f"Dataset de salida: {OUTPUT_DIR}")


# ## 1. Auditar imágenes y etiquetas YOLO
# 
# Se comprueban integridad, correspondencia imagen/etiqueta, clases, coordenadas y duplicados SHA-256. Las imágenes `healthy` deben tener etiquetas vacías y las `late_blight`, al menos una caja.

# In[ ]:


REPORTS_DIR = SOURCE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TOLERANCE = 1e-6


def image_class(image_name: str) -> str | None:
    return next((name for name in IMAGE_CLASSES if image_name.startswith(f"{name}_")), None)


def audit_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    images, boxes, errors = [], [], []

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_paths = {
            path.stem: path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_paths = {path.stem: path for path in label_dir.glob("*.txt")}

        for stem in sorted(set(image_paths) - set(label_paths)):
            errors.append((split, image_paths[stem].name, "falta etiqueta YOLO"))
        for stem in sorted(set(label_paths) - set(image_paths)):
            errors.append((split, label_paths[stem].name, "etiqueta sin imagen"))

        for stem, image_path in sorted(image_paths.items()):
            try:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("dimensiones no positivas")
            except Exception as error:
                errors.append((split, image_path.name, f"imagen inválida: {error}"))
                continue

            category = image_class(image_path.name)
            if category is None:
                errors.append((split, image_path.name, "clase de imagen desconocida"))
            images.append((split, image_path.name, category, width, height, digest))

            label_path = label_paths.get(stem)
            lines = [] if label_path is None else [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if category == "healthy" and lines:
                errors.append((split, label_path.name, "imagen healthy con cajas"))
            if category == "late_blight" and not lines:
                errors.append((split, image_path.name, "late_blight sin cajas"))

            seen = set()
            for line_number, line in enumerate(lines, 1):
                try:
                    values = tuple(map(float, line.split()))
                    if len(values) != 5:
                        raise ValueError("se esperaban 5 valores")
                    class_id, x_center, y_center, box_width, box_height = values
                    valid = (
                        class_id == 0
                        and all(np.isfinite(values))
                        and box_width > 0 and box_height > 0
                        and x_center - box_width / 2 >= -TOLERANCE
                        and y_center - box_height / 2 >= -TOLERANCE
                        and x_center + box_width / 2 <= 1 + TOLERANCE
                        and y_center + box_height / 2 <= 1 + TOLERANCE
                    )
                    if not valid:
                        raise ValueError("clase o caja fuera de límites")
                    key = tuple(round(value, 8) for value in values)
                    if key in seen:
                        raise ValueError("caja duplicada")
                    seen.add(key)
                except ValueError as error:
                    errors.append((split, label_path.name, f"línea {line_number}: {error}"))
                    continue

                boxes.append((
                    split, image_path.name, "late_blight",
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ))

    image_columns = ["split", "image_name", "image_class", "width", "height", "sha256"]
    box_columns = ["split", "image_name", "class_name", "xmin", "ymin", "xmax", "ymax"]
    error_columns = ["split", "file", "problem"]
    return (
        pd.DataFrame(images, columns=image_columns),
        pd.DataFrame(boxes, columns=box_columns),
        pd.DataFrame(errors, columns=error_columns),
    )


images_df, boxes_df, errors_df = audit_dataset(SOURCE_DIR)
duplicate_groups = images_df[images_df.duplicated("sha256", keep=False)].sort_values("sha256")
balance = pd.crosstab(images_df["split"], images_df["image_class"]).reindex(SPLITS, fill_value=0)

images_df.to_csv(REPORTS_DIR / "images.csv", index=False)
boxes_df.to_csv(REPORTS_DIR / "boxes.csv", index=False)
errors_df.to_csv(REPORTS_DIR / "validation_errors.csv", index=False)
duplicate_groups.to_csv(REPORTS_DIR / "duplicate_images.csv", index=False)
balance.to_csv(REPORTS_DIR / "class_balance.csv")

print(balance)
print(f"Imágenes: {len(images_df)} | Cajas: {len(boxes_df)} | Errores: {len(errors_df)}")
print(f"Copias duplicadas que se excluirán: {duplicate_groups.duplicated('sha256').sum()}")
assert errors_df.empty, "La auditoría encontró errores; revise reports/validation_errors.csv"


# ## 2. Balancear y exportar
# 
# Se conserva una sola imagen por hash, se igualan ambas clases al tamaño de la clase menor y se divide de forma estratificada en 70/20/10. A partir de las etiquetas YOLO se generan exportaciones equivalentes en COCO y Pascal VOC.

# In[12]:


def select_balanced_unique(images: pd.DataFrame) -> pd.DataFrame:
    unique = images.sort_values("image_name").drop_duplicates("sha256")
    class_size = int(unique.groupby("image_class").size().min())
    selected = []

    for category in IMAGE_CLASSES:
        group = unique[unique["image_class"] == category].sample(class_size, random_state=SEED)
        train_end = round(class_size * 0.70)
        val_end = train_end + round(class_size * 0.20)
        group = group.copy()
        group["validated_split"] = (
            ["train"] * train_end
            + ["val"] * (val_end - train_end)
            + ["test"] * (class_size - val_end)
        )
        selected.append(group)

    return pd.concat(selected, ignore_index=True)


def write_voc(xml_path: Path, image, image_boxes: pd.DataFrame) -> None:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image.image_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(int(image.width))
    ET.SubElement(size, "height").text = str(int(image.height))
    ET.SubElement(size, "depth").text = "3"

    for box in image_boxes.itertuples():
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "late_blight"
        bounds = ET.SubElement(obj, "bndbox")
        for tag, value in (
            ("xmin", max(0, round(box.xmin))),
            ("ymin", max(0, round(box.ymin))),
            ("xmax", min(int(image.width), round(box.xmax))),
            ("ymax", min(int(image.height), round(box.ymax))),
        ):
            ET.SubElement(bounds, tag).text = str(value)

    ET.ElementTree(annotation).write(xml_path, encoding="utf-8", xml_declaration=True)


def export_dataset(assignments: pd.DataFrame) -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    box_lookup = {name: group for name, group in boxes_df.groupby("image_name")}
    empty_boxes = boxes_df.iloc[0:0]

    for split in SPLITS:
        split_rows = assignments[assignments["validated_split"] == split]
        yolo_images = OUTPUT_DIR / "yolo" / "images" / split
        yolo_labels = OUTPUT_DIR / "yolo" / "labels" / split
        coco_images = OUTPUT_DIR / "coco" / split / "images"
        voc_images = OUTPUT_DIR / "pascal_voc" / split / "JPEGImages"
        voc_labels = OUTPUT_DIR / "pascal_voc" / split / "Annotations"
        for directory in (yolo_images, yolo_labels, coco_images, voc_images, voc_labels):
            directory.mkdir(parents=True, exist_ok=True)

        coco_image_rows, coco_box_rows = [], []
        annotation_id = 1
        for image_id, image in enumerate(split_rows.itertuples(), 1):
            source_image = SOURCE_DIR / "images" / image.split / image.image_name
            source_label = SOURCE_DIR / "labels" / image.split / f"{Path(image.image_name).stem}.txt"
            for destination in (yolo_images / image.image_name, coco_images / image.image_name, voc_images / image.image_name):
                shutil.copy2(source_image, destination)
            shutil.copy2(source_label, yolo_labels / source_label.name)

            image_boxes = box_lookup.get(image.image_name, empty_boxes)
            coco_image_rows.append({
                "id": image_id, "file_name": image.image_name,
                "width": int(image.width), "height": int(image.height),
            })
            for box in image_boxes.itertuples():
                width, height = box.xmax - box.xmin, box.ymax - box.ymin
                coco_box_rows.append({
                    "id": annotation_id, "image_id": image_id, "category_id": 1,
                    "bbox": [box.xmin, box.ymin, width, height],
                    "area": width * height, "iscrowd": 0,
                })
                annotation_id += 1
            write_voc(voc_labels / f"{Path(image.image_name).stem}.xml", image, image_boxes)

        coco = {
            "info": {"description": "Potato late blight detection dataset"},
            "images": coco_image_rows,
            "annotations": coco_box_rows,
            "categories": [{"id": 1, "name": "late_blight"}],
        }
        (OUTPUT_DIR / "coco" / split / "annotations.json").write_text(
            json.dumps(coco, indent=2), encoding="utf-8"
        )

    data_yaml = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: late_blight\n"
    (OUTPUT_DIR / "yolo" / "data.yaml").write_text(data_yaml, encoding="utf-8")


assignments = select_balanced_unique(images_df)
export_dataset(assignments)
assignments.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
final_balance = pd.crosstab(assignments["validated_split"], assignments["image_class"]).reindex(SPLITS)

print(final_balance)
print(f"Imágenes únicas exportadas: {len(assignments)}")
print(f"Hashes repetidos: {assignments['sha256'].duplicated().sum()}")


# ## 3. Validar y documentar los entregables
# 
# Se comparan imágenes y cajas entre los tres formatos, se verifica el balance y la ausencia de duplicados, y se generan la ficha técnica, los reportes, los ZIP y sus checksums SHA-256.

# In[13]:


def validate_exports(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        yolo_images = list((root / "yolo" / "images" / split).glob("*"))
        yolo_labels = list((root / "yolo" / "labels" / split).glob("*.txt"))
        yolo_boxes = sum(
            bool(line.strip())
            for path in yolo_labels
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        coco = json.loads((root / "coco" / split / "annotations.json").read_text(encoding="utf-8"))
        voc_images = list((root / "pascal_voc" / split / "JPEGImages").glob("*"))
        voc_xml = list((root / "pascal_voc" / split / "Annotations").glob("*.xml"))
        voc_boxes = sum(len(ET.parse(path).getroot().findall("object")) for path in voc_xml)

        image_counts = [len(yolo_images), len(yolo_labels), len(coco["images"]), len(voc_images), len(voc_xml)]
        box_counts = [yolo_boxes, len(coco["annotations"]), voc_boxes]
        rows.append({
            "split": split,
            "yolo_images": image_counts[0], "yolo_labels": image_counts[1], "yolo_boxes": box_counts[0],
            "coco_images": image_counts[2], "coco_boxes": box_counts[1],
            "voc_images": image_counts[3], "voc_xml": image_counts[4], "voc_boxes": box_counts[2],
            "consistent": len(set(image_counts)) == 1 and len(set(box_counts)) == 1,
        })
    return pd.DataFrame(rows)


validation = validate_exports(OUTPUT_DIR)
total_images = len(assignments)
total_boxes = int(validation["yolo_boxes"].sum())
class_size = int(final_balance.iloc[:, 0].sum())

assert validation["consistent"].all(), "Los formatos exportados no son consistentes"
assert assignments["sha256"].is_unique, "Hay imágenes duplicadas"
assert (final_balance["healthy"] == final_balance["late_blight"]).all(), "Las clases no están balanceadas"
assert int(validation["yolo_images"].sum()) == total_images

print(validation)

ficha = f"""# Ficha técnica: Potato Late Blight Detection Dataset

- **Estado:** dataset anotado, balanceado y exportado.
- **Objetivo:** detectar tizón tardío en hojas de papa.
- **Clases de imagen:** `healthy` y `late_blight`, {class_size} imágenes cada una.
- **Clase de objeto:** `late_blight` (ID 0); las imágenes `healthy` no contienen cajas.
- **Imágenes:** {total_images}.
- **Cajas:** {total_boxes}.
- **Particiones:** train {int(validation.loc[validation['split'] == 'train', 'yolo_images'].iloc[0])}, val {int(validation.loc[validation['split'] == 'val', 'yolo_images'].iloc[0])} y test {int(validation.loc[validation['split'] == 'test', 'yolo_images'].iloc[0])}.
- **Formatos:** YOLO, COCO y Pascal VOC.
- **Balance:** 50/50 en cada partición.
- **Validación:** imágenes y etiquetas consistentes, sin duplicados y con coordenadas válidas.
"""
(OUTPUT_DIR / "FICHA_TECNICA.md").write_text(ficha, encoding="utf-8")
validation.to_csv(OUTPUT_DIR / "VALIDATION_REPORT.csv", index=False)
final_balance.to_csv(OUTPUT_DIR / "CLASS_BALANCE.csv")

archives = []
for format_name in ("yolo", "coco", "pascal_voc"):
    archive = Path(shutil.make_archive(str(OUTPUT_DIR / format_name), "zip", OUTPUT_DIR / format_name))
    archives.append({
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    })
pd.DataFrame(archives).to_csv(OUTPUT_DIR / "CHECKSUMS.csv", index=False)

print(f"Validación superada: {total_images} imágenes, {total_boxes} cajas, 3 formatos.")
print(f"Entregables: {OUTPUT_DIR}")

