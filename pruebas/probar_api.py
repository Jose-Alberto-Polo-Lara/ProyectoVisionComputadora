"""
probar_api.py
=============


Pruebas automatizadas del API REST (src/api.py) usando el test client de
Flask, que invoca la aplicación WSGI en memoria sin necesidad de Docker ni
de un servidor gunicorn corriendo.


Permite validar el contrato del API (códigos de estado, forma de la
respuesta, manejo de errores) ANTES de construir la imagen Docker, algo
clave en este equipo donde Docker no está disponible.


Uso:
   pytest pruebas/probar_api.py -v
   python pruebas/probar_api.py          # ejecución directa (sin pytest)
"""


import io
import sys
from pathlib import Path


import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.api import app  # noqa: E402


EJEMPLOS_DIR = PROJECT_ROOT / "ejemplos"
IMAGEN_HEALTHY = EJEMPLOS_DIR / "test.jpg"       # etiqueta real conocida: healthy
IMAGEN_LATE_BLIGHT = EJEMPLOS_DIR / "test1.jpg"  # etiqueta real conocida: late_blight


CLASES_ESPERADAS = {"healthy", "late_blight"}




@pytest.fixture(scope="module")
def client():
   app.testing = True
   with app.test_client() as cliente:
       yield cliente




def test_health_responde_ok(client) -> None:
   respuesta = client.get("/health")
   assert respuesta.status_code == 200


   cuerpo = respuesta.get_json()
   assert cuerpo["estado"] == "ok"
   assert cuerpo["dispositivo"] in ("cpu", "cuda", "mps")
   assert set(cuerpo["clases"]) == CLASES_ESPERADAS




def test_predict_clasifica_imagen_healthy(client) -> None:
   with open(IMAGEN_HEALTHY, "rb") as f:
       respuesta = client.post(
           "/predict",
           data={"archivo": (io.BytesIO(f.read()), IMAGEN_HEALTHY.name)},
           content_type="multipart/form-data",
       )


   assert respuesta.status_code == 200
   cuerpo = respuesta.get_json()
   assert cuerpo["imagen"] == IMAGEN_HEALTHY.name
   assert cuerpo["clase"] == "healthy"
   assert 0 <= cuerpo["confianza_pct"] <= 100
   assert set(cuerpo["probabilidades"].keys()) == CLASES_ESPERADAS
   assert cuerpo["latencia_ms"] >= 0




def test_predict_clasifica_imagen_late_blight(client) -> None:
   with open(IMAGEN_LATE_BLIGHT, "rb") as f:
       respuesta = client.post(
           "/predict",
           data={"archivo": (io.BytesIO(f.read()), IMAGEN_LATE_BLIGHT.name)},
           content_type="multipart/form-data",
       )


   assert respuesta.status_code == 200
   cuerpo = respuesta.get_json()
   assert cuerpo["clase"] == "late_blight"




def test_predict_rechaza_formato_no_soportado(client) -> None:
   contenido_falso = b"esto no es una imagen"
   respuesta = client.post(
       "/predict",
       data={"archivo": (io.BytesIO(contenido_falso), "documento.txt")},
       content_type="multipart/form-data",
   )


   assert respuesta.status_code == 400
   assert "Formato no soportado" in respuesta.get_json()["detail"]




def test_predict_rechaza_imagen_corrupta(client) -> None:
   contenido_falso = b"bytes-invalidos-no-son-un-jpg"
   respuesta = client.post(
       "/predict",
       data={"archivo": (io.BytesIO(contenido_falso), "falsa.jpg")},
       content_type="multipart/form-data",
   )


   assert respuesta.status_code == 400
   assert "no es una imagen válida" in respuesta.get_json()["detail"]




def test_predict_batch_clasifica_varias_imagenes(client) -> None:
   with open(IMAGEN_HEALTHY, "rb") as f1, open(IMAGEN_LATE_BLIGHT, "rb") as f2:
       respuesta = client.post(
           "/predict/batch",
           data={
               "archivos": [
                   (io.BytesIO(f1.read()), IMAGEN_HEALTHY.name),
                   (io.BytesIO(f2.read()), IMAGEN_LATE_BLIGHT.name),
               ]
           },
           content_type="multipart/form-data",
       )


   assert respuesta.status_code == 200
   cuerpo = respuesta.get_json()
   resultados = cuerpo["resultados"]
   assert len(resultados) == 2


   por_imagen = {r["imagen"]: r for r in resultados}
   assert por_imagen[IMAGEN_HEALTHY.name]["clase"] == "healthy"
   assert por_imagen[IMAGEN_LATE_BLIGHT.name]["clase"] == "late_blight"




def test_predict_batch_rechaza_lista_vacia(client) -> None:
   respuesta = client.post("/predict/batch", data={}, content_type="multipart/form-data")
   # Flask responde 422 (definido explícitamente en el endpoint) cuando no llega el campo "archivos"
   assert respuesta.status_code == 422




if __name__ == "__main__":
   sys.exit(pytest.main([__file__, "-v"]))



