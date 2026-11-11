"""
probar_clasificacion.py
========================

Prueba que el modelo entrenado (models/model.pt) clasifica correctamente las
imágenes de hojas de papa. Dos niveles de verificación:

    1. Ejemplos conocidos (ejemplos/): test.jpg es "healthy" y test1.jpg es
       "late_blight" (según la documentación del notebook de entrenamiento).
    2. Partición de test del dataset validado
       (data/potato_late_blight_dataset_validated/yolo/images/test), cuya
       etiqueta real se infiere del prefijo del nombre de archivo
       (healthy_* / late_blight_*): se calcula el accuracy sobre las 58
       imágenes de test y se exige que supere un umbral mínimo.

Uso:
    python pruebas/probar_clasificacion.py
    python pruebas/probar_clasificacion.py --umbral 0.90 --modelo models/model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELO_DEFECTO = PROJECT_ROOT / "models" / "model.pt"
CLASES_DEFECTO = PROJECT_ROOT / "config" / "clases.txt"
EJEMPLOS_DIR = PROJECT_ROOT / "ejemplos"
TEST_DIR = PROJECT_ROOT / "data" / "potato_late_blight_dataset_validated" / "yolo" / "images" / "test"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
])

# Etiquetas reales conocidas para las imágenes de ejemplo (documentadas en el
# notebook de entrenamiento: "usa test.jpg (healthy) o test1.jpg (late_blight)")
ETIQUETAS_EJEMPLOS_CONOCIDAS = {
    "test.jpg": "healthy",
    "test1.jpg": "late_blight",
}


def seleccionar_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_class_desde_nombre(filename: str) -> str:
    return "late_blight" if filename.startswith("late_blight_") else "healthy"


def cargar_modelo(ruta_modelo: Path, n_clases: int, dispositivo: torch.device) -> nn.Module:
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)
    estado = torch.load(ruta_modelo, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


def predecir(modelo: nn.Module, ruta_imagen: Path, clases: list[str], dispositivo: torch.device) -> tuple[str, float]:
    imagen = Image.open(ruta_imagen).convert("RGB")
    tensor = TRANSFORM(imagen).unsqueeze(0).to(dispositivo)
    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.softmax(logits, dim=1)[0]
        confianza, clase_id = torch.max(probabilidades, dim=0)
    return clases[clase_id.item()], confianza.item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de clasificación correcta de imágenes")
    parser.add_argument("--modelo", type=str, default=str(MODELO_DEFECTO))
    parser.add_argument("--clases", type=str, default=str(CLASES_DEFECTO))
    parser.add_argument("--umbral", type=float, default=0.90, help="Accuracy mínimo exigido sobre el split de test")
    args = parser.parse_args()

    ruta_modelo = Path(args.modelo)
    if not ruta_modelo.exists():
        print(f"Error: no se encuentra el modelo en {ruta_modelo}", file=sys.stderr)
        return 1

    clases = [c.strip() for c in Path(args.clases).read_text(encoding="utf-8").splitlines() if c.strip()]
    dispositivo = seleccionar_dispositivo()
    modelo = cargar_modelo(ruta_modelo, len(clases), dispositivo)
    print(f"Modelo cargado desde {ruta_modelo} (dispositivo={dispositivo})")

    exito_general = True

    # --- 1. Ejemplos con etiqueta conocida ---
    print("\n== Verificación 1: ejemplos con etiqueta conocida ==")
    resultados_ejemplos = []
    for nombre_archivo, etiqueta_esperada in ETIQUETAS_EJEMPLOS_CONOCIDAS.items():
        ruta = EJEMPLOS_DIR / nombre_archivo
        if not ruta.exists():
            print(f"  [OMITIDO] {nombre_archivo} no existe en {EJEMPLOS_DIR}")
            continue
        clase_predicha, confianza = predecir(modelo, ruta, clases, dispositivo)
        acierto = clase_predicha == etiqueta_esperada
        exito_general &= acierto
        estado = "OK" if acierto else "FALLÓ"
        print(f"  [{estado}] {nombre_archivo}: esperado={etiqueta_esperada} "
              f"predicho={clase_predicha} (confianza={confianza * 100:.2f}%)")
        resultados_ejemplos.append({
            "imagen": nombre_archivo, "esperado": etiqueta_esperada,
            "predicho": clase_predicha, "confianza_pct": round(confianza * 100, 2), "acierto": acierto,
        })

    # --- 2. Partición de test completa del dataset ---
    print("\n== Verificación 2: partición de test del dataset ==")
    if not TEST_DIR.is_dir():
        print(f"  [OMITIDO] No se encontró {TEST_DIR}. "
              "Ejecuta primero preparacion_dataset/dataset_tizon_papa.py.")
        accuracy_test = None
        resultados_test = []
    else:
        correctos = 0
        resultados_test = []
        imagenes_test = sorted(
            p for p in TEST_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for ruta_imagen in imagenes_test:
            etiqueta_real = infer_class_desde_nombre(ruta_imagen.name)
            clase_predicha, confianza = predecir(modelo, ruta_imagen, clases, dispositivo)
            acierto = clase_predicha == etiqueta_real
            correctos += int(acierto)
            resultados_test.append({
                "imagen": ruta_imagen.name, "real": etiqueta_real,
                "predicho": clase_predicha, "confianza_pct": round(confianza * 100, 2), "acierto": acierto,
            })

        total = len(imagenes_test)
        accuracy_test = correctos / total if total else 0.0
        print(f"  Accuracy sobre {total} imágenes de test: {accuracy_test * 100:.2f}% "
              f"({correctos}/{total} correctas)")

        fallidas = [r for r in resultados_test if not r["acierto"]]
        if fallidas:
            print(f"  Imágenes mal clasificadas ({len(fallidas)}):")
            for r in fallidas:
                print(f"    - {r['imagen']}: real={r['real']} predicho={r['predicho']} "
                      f"(confianza={r['confianza_pct']}%)")

        if accuracy_test < args.umbral:
            exito_general = False
            print(f"  [FALLÓ] Accuracy {accuracy_test * 100:.2f}% por debajo del umbral "
                  f"exigido {args.umbral * 100:.0f}%.")
        else:
            print(f"  [OK] Accuracy por encima del umbral exigido ({args.umbral * 100:.0f}%).")

    # --- Persistir reporte ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reporte = {
        "modelo": str(ruta_modelo),
        "dispositivo": str(dispositivo),
        "ejemplos_conocidos": resultados_ejemplos,
        "accuracy_test": accuracy_test,
        "umbral_exigido": args.umbral,
        "resultados_test": resultados_test,
        "exito_general": bool(exito_general),
    }
    ruta_reporte = OUTPUTS_DIR / "reporte_clasificacion.json"
    ruta_reporte.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en: {ruta_reporte}")

    if exito_general:
        print("\nRESULTADO FINAL: las imágenes se están clasificando correctamente.")
        return 0

    print("\nRESULTADO FINAL: hay fallos en la clasificación. Revisa el reporte.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
