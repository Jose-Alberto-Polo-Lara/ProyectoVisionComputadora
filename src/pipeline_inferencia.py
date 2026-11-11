"""
pipeline_inferencia.py
======================

Pipeline operativo de inferencia para el clasificador de tizón tardío en hojas
de papa (healthy / late_blight), siguiendo las fases descritas en el material
"Unidad 4 - Tema 4.1: Pipeline Operativo":

    Fase 1  Preparación del modelo entrenado   -> cargar_modelo()
    Fase 2  Configuración del entorno          -> requirements.txt (carpeta raíz del proyecto)
    Fase 3  Ingesta de datos (Input Layer)     -> resolver_entradas() + validaciones de formato
    Fase 4  Preprocesamiento                   -> preprocesar_imagen()
    Fase 5  Carga del modelo                   -> cargar_modelo() + prueba de integridad
    Fase 6  Inferencia                         -> ejecutar_inferencia()
    Fase 7  Posprocesamiento                   -> ejecutar_inferencia() (softmax + clase + confianza)
    Fase 8  Generación de salida                -> mostrar e, opcionalmente, persistir en JSON/CSV
    Fase 9  Registro, logs y control de errores -> módulo logging (logs/pipeline.log)
    Fase 10 Estructura de carpetas              -> src/, models/, config/, outputs/, logs/

A diferencia del ejemplo genérico del material (TensorFlow/Keras + .h5 + cv2),
este pipeline usa PyTorch porque así fue entrenado el modelo real
(`model.pt`, ResNet-18 con transfer learning, ver
"Image classification with CNN modelo vision.ipynb" y models/model_info.json).
El preprocesamiento usa PIL + torchvision.transforms (no cv2) para ser
EXACTAMENTE idéntico al usado durante el entrenamiento.

Uso:
    python src/pipeline_inferencia.py --imagen ejemplos/test.jpg
    python src/pipeline_inferencia.py --directorio ejemplos/
    python src/pipeline_inferencia.py --imagen ejemplos/test.jpg --salida outputs/resultado.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a la raíz del proyecto "proyecto vision computadora")
# ---------------------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
MODELO_DEFECTO = RAIZ_PROYECTO / "models" / "model.pt"
CLASES_DEFECTO = RAIZ_PROYECTO / "config" / "clases.txt"
LOGS_DIR = RAIZ_PROYECTO / "logs"
OUTPUTS_DIR = RAIZ_PROYECTO / "outputs"

FORMATOS_VALIDOS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Transformación IDÉNTICA a la usada en el entrenamiento (ver model_info.json)
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
TAMANO_ENTRADA = (224, 224)

TRANSFORM_INFERENCIA = transforms.Compose([
    transforms.Resize(TAMANO_ENTRADA),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
])


# ---------------------------------------------------------------------------
# Fase 9: Registro y logs
# ---------------------------------------------------------------------------
def configurar_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline_tizon_papa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    manejador_archivo = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    manejador_archivo.setFormatter(formato)
    logger.addHandler(manejador_archivo)

    manejador_consola = logging.StreamHandler(sys.stdout)
    manejador_consola.setFormatter(formato)
    logger.addHandler(manejador_consola)

    return logger


logger = configurar_logging()


# ---------------------------------------------------------------------------
# Fase 5: Carga del modelo (con validación e integridad)
# ---------------------------------------------------------------------------
def seleccionar_dispositivo(preferencia: str = "auto") -> torch.device:
    """Selecciona CPU/GPU/MPS según disponibilidad (Fase 5: selección de dispositivo)."""
    if preferencia != "auto":
        return torch.device(preferencia)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cargar_clases(ruta_clases: Path) -> list[str]:
    if not ruta_clases.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de clases en: {ruta_clases}")
    with open(ruta_clases, "r", encoding="utf-8") as f:
        clases = [linea.strip() for linea in f.readlines() if linea.strip()]
    if not clases:
        raise ValueError(f"El archivo de clases {ruta_clases} está vacío.")
    return clases


def cargar_modelo(ruta_modelo: Path, n_clases: int, dispositivo: torch.device) -> nn.Module:
    """Reconstruye la arquitectura ResNet-18 + capa de salida y carga los pesos
    entrenados (state_dict). Incluye prueba de integridad con un forward-pass
    dummy, tal como pide la Fase 5 del pipeline operativo."""
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en: {ruta_modelo}")

    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)

    try:
        estado = torch.load(ruta_modelo, map_location=dispositivo, weights_only=True)
    except Exception as exc:  # state_dict corrupto o incompatible
        raise RuntimeError(f"No se pudo cargar el state_dict desde {ruta_modelo}: {exc}") from exc

    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()

    # Prueba de integridad: inferencia dummy para confirmar que el modelo responde
    with torch.no_grad():
        entrada_prueba = torch.zeros(1, 3, *TAMANO_ENTRADA, device=dispositivo)
        salida_prueba = modelo(entrada_prueba)
        if salida_prueba.shape[-1] != n_clases:
            raise RuntimeError(
                f"El modelo produce {salida_prueba.shape[-1]} salidas pero se "
                f"esperaban {n_clases} (según {CLASES_DEFECTO.name})."
            )

    logger.info(f"Modelo cargado correctamente desde {ruta_modelo} (dispositivo={dispositivo})")
    return modelo


# ---------------------------------------------------------------------------
# Fase 3: Ingesta de datos (Input Layer)
# ---------------------------------------------------------------------------
def resolver_entradas(imagen: str | None, directorio: str | None) -> list[Path]:
    """Resuelve la lista de imágenes a procesar a partir de una imagen individual
    o un directorio completo, validando rutas y formatos permitidos."""
    rutas: list[Path] = []

    if imagen:
        ruta = Path(imagen)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encuentra la imagen: {ruta}")
        if ruta.suffix.lower() not in FORMATOS_VALIDOS:
            raise ValueError(f"Formato no soportado '{ruta.suffix}'. Formatos válidos: {FORMATOS_VALIDOS}")
        rutas.append(ruta)

    if directorio:
        ruta_dir = Path(directorio)
        if not ruta_dir.is_dir():
            raise FileNotFoundError(f"No se encuentra el directorio: {ruta_dir}")
        encontradas = sorted(
            p for p in ruta_dir.iterdir() if p.suffix.lower() in FORMATOS_VALIDOS
        )
        if not encontradas:
            raise FileNotFoundError(f"No hay imágenes con formato válido en: {ruta_dir}")
        rutas.extend(encontradas)

    if not rutas:
        raise ValueError("Debes especificar --imagen o --directorio con al menos una imagen válida.")

    return rutas


# ---------------------------------------------------------------------------
# Fase 4: Preprocesamiento (idéntico al entrenamiento)
# ---------------------------------------------------------------------------
def preprocesar_imagen(ruta_imagen: Path) -> tuple[torch.Tensor, Image.Image]:
    try:
        imagen_pil = Image.open(ruta_imagen).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"El archivo no es una imagen válida o está corrupto: {ruta_imagen}") from exc

    tensor = TRANSFORM_INFERENCIA(imagen_pil).unsqueeze(0)  # añade dimensión de batch
    return tensor, imagen_pil


# ---------------------------------------------------------------------------
# Fase 6 y 7: Inferencia + Posprocesamiento
# ---------------------------------------------------------------------------
def ejecutar_inferencia(modelo: nn.Module, tensor: torch.Tensor, clases: list[str],
                         dispositivo: torch.device) -> dict:
    """Inferencia de una sola imagen (tensor con dimensión de batch=1)."""
    tensor = tensor.to(dispositivo)
    inicio = time.time()
    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.softmax(logits, dim=1)[0]
        confianza, clase_id = torch.max(probabilidades, dim=0)
    latencia_ms = (time.time() - inicio) * 1000

    return {
        "clase": clases[clase_id.item()],
        "confianza_pct": round(confianza.item() * 100, 2),
        "probabilidades": {c: round(p.item() * 100, 2) for c, p in zip(clases, probabilidades)},
        "latencia_ms": round(latencia_ms, 2),
    }


def ejecutar_inferencia_lote(modelo: nn.Module, tensores: list[torch.Tensor], clases: list[str],
                              dispositivo: torch.device) -> list[dict]:
    """Inferencia POR LOTE: apila varias imágenes ya preprocesadas en un único
    tensor [N, C, H, W] y realiza un solo forward-pass para todo el lote
    (más eficiente que procesar una por una, especialmente en GPU)."""
    lote = torch.cat(tensores, dim=0).to(dispositivo)  # [N, 3, 224, 224]
    inicio = time.time()
    with torch.no_grad():
        logits = modelo(lote)
        probabilidades = torch.softmax(logits, dim=1)
        confianzas, clases_id = torch.max(probabilidades, dim=1)
    latencia_total_ms = (time.time() - inicio) * 1000
    latencia_por_imagen_ms = latencia_total_ms / len(tensores)

    resultados = []
    for i in range(len(tensores)):
        resultados.append({
            "clase": clases[clases_id[i].item()],
            "confianza_pct": round(confianzas[i].item() * 100, 2),
            "probabilidades": {c: round(p.item() * 100, 2) for c, p in zip(clases, probabilidades[i])},
            "latencia_ms": round(latencia_por_imagen_ms, 2),
        })
    logger.info(f"Inferencia por lote: {len(tensores)} imágenes en {latencia_total_ms:.2f} ms "
                f"({latencia_por_imagen_ms:.2f} ms/imagen)")
    return resultados


# ---------------------------------------------------------------------------
# Fase 8: Generación de salida
# ---------------------------------------------------------------------------
def mostrar_resultado(ruta_imagen: Path, resultado: dict) -> None:
    print(f"\n{'=' * 60}")
    print("RESULTADO DE INFERENCIA")
    print(f"Imagen       : {ruta_imagen.name}")
    print(f"Predicción   : {resultado['clase']}")
    print(f"Confianza    : {resultado['confianza_pct']}%")
    print(f"Probabilidades: {resultado['probabilidades']}")
    print(f"Latencia     : {resultado['latencia_ms']} ms")
    print(f"{'=' * 60}")


def mostrar_resultado_tiempo_real(indice: int, ruta_imagen: Path, resultado: dict) -> None:
    marca_tiempo = time.strftime("%H:%M:%S")
    print(f"[{marca_tiempo}] Frame #{indice:03d} | {ruta_imagen.name:30s} "
          f"-> {resultado['clase']:12s} ({resultado['confianza_pct']}%) "
          f"| latencia={resultado['latencia_ms']} ms")


def ejecutar_tiempo_real_simulado(modelo: nn.Module, rutas_imagenes: list[Path], clases: list[str],
                                   dispositivo: torch.device, intervalo_segundos: float) -> list[dict]:
    """Simula la llegada de imágenes en tiempo real (p. ej. una cámara o un
    stream), procesando una por una con un retardo fijo entre "frames" y
    mostrando cada resultado apenas está disponible, en vez de esperar a
    tener todo el lote completo."""
    resultados_completos = []
    print(f"\nIniciando inferencia en tiempo real simulado "
          f"(intervalo={intervalo_segundos}s entre frames)...")
    for indice, ruta_imagen in enumerate(rutas_imagenes, start=1):
        try:
            tensor, _ = preprocesar_imagen(ruta_imagen)
            resultado = ejecutar_inferencia(modelo, tensor, clases, dispositivo)
            mostrar_resultado_tiempo_real(indice, ruta_imagen, resultado)
            logger.info(
                f"TIEMPO_REAL | frame={indice} | {ruta_imagen.name} -> {resultado['clase']} "
                f"({resultado['confianza_pct']}%, {resultado['latencia_ms']} ms)"
            )
            resultados_completos.append({"imagen": str(ruta_imagen), **resultado})
        except (ValueError, OSError) as exc:
            logger.error(f"Fallo al procesar frame simulado {ruta_imagen}: {exc}")
        if indice < len(rutas_imagenes):
            time.sleep(intervalo_segundos)
    return resultados_completos


def persistir_resultados(resultados: list[dict], ruta_salida: Path) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if ruta_salida.suffix.lower() == ".json":
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
    elif ruta_salida.suffix.lower() == ".csv":
        campos = ["imagen", "clase", "confianza_pct", "latencia_ms"]
        with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=campos)
            escritor.writeheader()
            for r in resultados:
                escritor.writerow({k: r[k] for k in campos})
    else:
        raise ValueError("La extensión de --salida debe ser .json o .csv")
    logger.info(f"Resultados persistidos en: {ruta_salida}")


# ---------------------------------------------------------------------------
# Orquestación principal
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline de inferencia para el clasificador de tizón tardío en hojas de papa"
    )
    parser.add_argument("--imagen", type=str, default=None, help="Ruta a una imagen individual a predecir")
    parser.add_argument("--directorio", type=str, default=None, help="Ruta a un directorio de imágenes a predecir")
    parser.add_argument("--modelo", type=str, default=str(MODELO_DEFECTO), help="Ruta al modelo entrenado (.pt)")
    parser.add_argument("--clases", type=str, default=str(CLASES_DEFECTO), help="Archivo con los nombres de las clases")
    parser.add_argument("--dispositivo", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                         help="Dispositivo de ejecución")
    parser.add_argument("--salida", type=str, default=None, help="Ruta opcional para persistir resultados (.json o .csv)")
    parser.add_argument("--modo", type=str, default="lote", choices=["lote", "tiempo-real"],
                         help="'lote': une todas las imágenes en un único forward-pass. "
                              "'tiempo-real': procesa una imagen a la vez simulando un stream en vivo")
    parser.add_argument("--intervalo", type=float, default=1.0,
                         help="Segundos de espera entre frames en modo --modo tiempo-real")
    args = parser.parse_args()

    try:
        dispositivo = seleccionar_dispositivo(args.dispositivo)
        clases = cargar_clases(Path(args.clases))
        modelo = cargar_modelo(Path(args.modelo), n_clases=len(clases), dispositivo=dispositivo)
        rutas_imagenes = resolver_entradas(args.imagen, args.directorio)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"Error de inicialización del pipeline: {exc}")
        return 1

    if args.modo == "tiempo-real":
        resultados_completos = ejecutar_tiempo_real_simulado(
            modelo, rutas_imagenes, clases, dispositivo, args.intervalo
        )
    else:
        tensores, rutas_validas, resultados_completos = [], [], []
        for ruta_imagen in rutas_imagenes:
            try:
                tensor, _ = preprocesar_imagen(ruta_imagen)
                tensores.append(tensor)
                rutas_validas.append(ruta_imagen)
            except (ValueError, OSError) as exc:
                logger.error(f"Fallo al preprocesar {ruta_imagen}: {exc}")

        if tensores:
            resultados_lote = ejecutar_inferencia_lote(modelo, tensores, clases, dispositivo)
            for ruta_imagen, resultado in zip(rutas_validas, resultados_lote):
                mostrar_resultado(ruta_imagen, resultado)
                logger.info(
                    f"OK | {ruta_imagen.name} -> {resultado['clase']} "
                    f"({resultado['confianza_pct']}%, {resultado['latencia_ms']} ms)"
                )
                resultados_completos.append({"imagen": str(ruta_imagen), **resultado})

    if args.salida and resultados_completos:
        persistir_resultados(resultados_completos, Path(args.salida))

    return 0 if resultados_completos else 1


if __name__ == "__main__":
    sys.exit(main())
