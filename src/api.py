"""
api.py
======


API REST (Flask) que expone el modelo de clasificación de tizón tardío en
hojas de papa (healthy / late_blight) para ser consumido desde Docker.


Reutiliza exactamente las mismas funciones del pipeline operativo
(`pipeline_inferencia.py`): carga de modelo, preprocesamiento, inferencia y
selección de dispositivo, de modo que el comportamiento es idéntico al CLI.


Endpoints:
   GET  /health            -> estado del servicio y metadata del modelo
   POST /predict           -> clasifica una única imagen (multipart/form-data, campo "archivo")
   POST /predict/batch     -> clasifica varias imágenes en un solo forward-pass (campo "archivos")


Ejecución local (desarrollo):
   flask --app src.api run --host 0.0.0.0 --port 8000


Ejecución en producción / Docker: ver Dockerfile (CMD con gunicorn).
"""


import io
from pathlib import Path


from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError


from .pipeline_inferencia import (
   CLASES_DEFECTO,
   FORMATOS_VALIDOS,
   MODELO_DEFECTO,
   TRANSFORM_INFERENCIA,
   cargar_clases,
   cargar_modelo,
   ejecutar_inferencia,
   ejecutar_inferencia_lote,
   logger,
   seleccionar_dispositivo,
)


app = Flask(__name__)


# ---------------------------------------------------------------------------
# Carga del modelo una sola vez al importar el módulo (Fase 5: carga del modelo).
# Con Flask + gunicorn, cada worker importa este módulo una vez al arrancar,
# por lo que el modelo queda cargado en memoria antes de atender requests.
# ---------------------------------------------------------------------------
_dispositivo = seleccionar_dispositivo("auto")
_clases = cargar_clases(CLASES_DEFECTO)
_modelo = cargar_modelo(MODELO_DEFECTO, n_clases=len(_clases), dispositivo=_dispositivo)
logger.info(f"API lista. Dispositivo={_dispositivo}, clases={_clases}")




def _validar_y_abrir_imagen(nombre: str, contenido: bytes):
   """Devuelve (imagen_pil, None) si es válida, o (None, (respuesta_json, codigo)) si falla."""
   extension = Path(nombre).suffix.lower()
   if extension not in FORMATOS_VALIDOS:
       return None, (
           jsonify({"detail": f"Formato no soportado '{extension}'. Formatos válidos: {FORMATOS_VALIDOS}"}),
           400,
       )
   try:
       return Image.open(io.BytesIO(contenido)).convert("RGB"), None
   except UnidentifiedImageError:
       return None, (jsonify({"detail": f"El archivo '{nombre}' no es una imagen válida."}), 400)




@app.get("/health")
def health():
   return jsonify({"estado": "ok", "dispositivo": str(_dispositivo), "clases": _clases})




@app.post("/predict")
def predict():
   if "archivo" not in request.files:
       return jsonify({"detail": "Debes enviar un archivo en el campo 'archivo'."}), 422


   archivo = request.files["archivo"]
   contenido = archivo.read()
   imagen_pil, error = _validar_y_abrir_imagen(archivo.filename, contenido)
   if error is not None:
       return error


   tensor = TRANSFORM_INFERENCIA(imagen_pil).unsqueeze(0)
   resultado = ejecutar_inferencia(_modelo, tensor, _clases, _dispositivo)
   logger.info(f"API /predict | {archivo.filename} -> {resultado['clase']} ({resultado['confianza_pct']}%)")
   return jsonify({"imagen": archivo.filename, **resultado})




@app.post("/predict/batch")
def predict_batch():
   archivos = request.files.getlist("archivos")
   if not archivos:
       return jsonify({"detail": "Debes enviar al menos un archivo en el campo 'archivos'."}), 422


   tensores, nombres_validos = [], []
   for archivo in archivos:
       contenido = archivo.read()
       imagen_pil, error = _validar_y_abrir_imagen(archivo.filename, contenido)
       if error is not None:
           return error
       tensores.append(TRANSFORM_INFERENCIA(imagen_pil).unsqueeze(0))
       nombres_validos.append(archivo.filename)


   resultados = ejecutar_inferencia_lote(_modelo, tensores, _clases, _dispositivo)
   respuesta = [{"imagen": nombre, **resultado} for nombre, resultado in zip(nombres_validos, resultados)]
   logger.info(f"API /predict/batch | {len(respuesta)} imágenes procesadas")
   return jsonify({"resultados": respuesta})




if __name__ == "__main__":
   app.run(host="0.0.0.0", port=8000)

