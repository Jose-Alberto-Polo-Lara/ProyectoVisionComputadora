"""
ejecutar_pipeline_completo.py
==============================

Rutina orquestadora que ejecuta, en orden, todo el ciclo del proyecto:

    1. Generar/actualizar el dataset validado a partir de las anotaciones
       crudas (preparacion_dataset/dataset_tizon_papa.py).
    2. Entrenar (o re-entrenar) el modelo de clasificación
       (entrenamiento/entrenar_modelo.py).
    3. Probar que las imágenes se clasifican correctamente
       (pruebas/probar_clasificacion.py).

Se detiene en el primer paso que falle (código de salida != 0).

Uso:
    python ejecutar_pipeline_completo.py
    python ejecutar_pipeline_completo.py --saltar-dataset --saltar-entrenamiento
    python ejecutar_pipeline_completo.py --epocas 5 --umbral 0.85
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def ejecutar(descripcion: str, comando: list[str], cwd: Path) -> int:
    print(f"\n{'=' * 70}\n{descripcion}\n{'=' * 70}")
    resultado = subprocess.run(comando, cwd=cwd)
    return resultado.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo del proyecto")
    parser.add_argument("--saltar-dataset", action="store_true", help="No regenerar el dataset")
    parser.add_argument("--saltar-entrenamiento", action="store_true", help="No reentrenar el modelo")
    parser.add_argument("--saltar-pruebas", action="store_true", help="No ejecutar la prueba de clasificación")
    parser.add_argument("--epocas", type=int, default=10, help="Épocas de entrenamiento")
    parser.add_argument("--umbral", type=float, default=0.90, help="Accuracy mínimo exigido en la prueba")
    args = parser.parse_args()

    if not args.saltar_dataset:
        codigo = ejecutar(
            "Paso 1/3: Generar dataset (preparacion_dataset/dataset_tizon_papa.py)",
            [sys.executable, str(PROJECT_ROOT / "preparacion_dataset" / "dataset_tizon_papa.py")],
            cwd=PROJECT_ROOT / "data",
        )
        if codigo != 0:
            print("Fallo al generar el dataset. Deteniendo pipeline.", file=sys.stderr)
            return codigo

    if not args.saltar_entrenamiento:
        codigo = ejecutar(
            "Paso 2/3: Entrenar modelo (entrenamiento/entrenar_modelo.py)",
            [sys.executable, str(PROJECT_ROOT / "entrenamiento" / "entrenar_modelo.py"),
             "--epocas", str(args.epocas)],
            cwd=PROJECT_ROOT,
        )
        if codigo != 0:
            print("Fallo al entrenar el modelo. Deteniendo pipeline.", file=sys.stderr)
            return codigo

    if not args.saltar_pruebas:
        codigo = ejecutar(
            "Paso 3/3: Probar clasificación correcta (pruebas/probar_clasificacion.py)",
            [sys.executable, str(PROJECT_ROOT / "pruebas" / "probar_clasificacion.py"),
             "--umbral", str(args.umbral)],
            cwd=PROJECT_ROOT,
        )
        if codigo != 0:
            print("La prueba de clasificación falló.", file=sys.stderr)
            return codigo

    print("\nPipeline completo ejecutado con éxito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
