# Imagen base liviana con Python 3.12 (mismo intérprete usado para entrenar/probar el modelo)
FROM python:3.12-slim


# Evita archivos .pyc y fuerza salida sin buffer (logs visibles en tiempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
   PYTHONUNBUFFERED=1 \
   PIP_NO_CACHE_DIR=1


WORKDIR /app


# Dependencias de sistema necesarias para Pillow/torchvision (decodificación de imágenes)
RUN apt-get update && apt-get install -y --no-install-recommends \
   libjpeg62-turbo \
   zlib1g \
   && rm -rf /var/lib/apt/lists/*


# Se instalan primero las dependencias para aprovechar la cache de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Se copia únicamente lo necesario para servir el modelo (no se copian data/ ni dataset/)
COPY src/ src/
COPY models/ models/
COPY config/ config/


# Carpetas de artefactos en tiempo de ejecución (se pueden montar como volúmenes)
RUN mkdir -p logs outputs


EXPOSE 8000


# Verifica que el servicio responde antes de considerarlo "saludable"
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1


CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "src.api:app"]



