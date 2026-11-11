# Proyecto Visión Computadora — Pipeline Operativo: Tizón Tardío en Hojas de Papa

Este proyecto operacionaliza el modelo entrenado en
`Image classification with CNN modelo vision.ipynb` (ResNet-18, transfer
learning, clasificación binaria `healthy` vs `late_blight`) siguiendo la
estructura de **Pipeline Operativo** del material "Unidad 4 - Tema 4.1".

Además, se centraliza aquí todo el proceso previo del proyecto: la
preparación/auditoría del dataset (`preparacion_dataset/dataset_tizon_papa.py`,
convertido desde `dataset_tizon_papa.ipynb`) y el dataset resultante
(`data/`), para que el ciclo completo —desde las anotaciones originales hasta
el modelo entrenado y su pipeline de inferencia— quede en un solo lugar.

## 1. Validación de preparación (checklist por fase)

| Fase PDF | Requisito | Estado | Evidencia |
|---|---|---|---|
| 1. Preparación del modelo | Modelo entrenado, versionado, con métricas y fecha documentadas | ✅ Listo | `models/model.pt` + `models/model_info.json` (val_accuracy máx. 99.15%, ver `training_metrics_tizon_papa.csv` y `REPORTE-CLASIFICACION-TIZON-PAPA.md`) |
| 1. Optimización (opcional) | Cuantización/pruning | ⚠️ No aplicado | No es obligatorio por el PDF; el modelo es ligero (ResNet-18) y no se requiere para los volúmenes de este proyecto |
| 2. Configuración del entorno | Dependencias declaradas y reproducibles | ✅ Listo | `requirements.txt` (torch, torchvision, Pillow, numpy con versiones fijas) |
| 3. Ingesta de datos | Soporte de imagen individual y lote, validación de formato/ruta | ✅ Listo | `src/pipeline_inferencia.py::resolver_entradas()` (soporta `--imagen` y `--directorio`, valida extensiones) |
| 4. Preprocesamiento | Idéntico al usado en entrenamiento | ✅ Listo | `TRANSFORM_INFERENCIA` replica exactamente Resize(224,224) + ToTensor + Normalize ImageNet del notebook de entrenamiento |
| 5. Carga del modelo | Reconstrucción de arquitectura, selección de dispositivo, prueba de integridad | ✅ Listo | `cargar_modelo()` reconstruye ResNet-18 + `fc` de 2 salidas, soporta cpu/cuda/mps, ejecuta un forward-pass dummy de verificación |
| 6. Inferencia | Ejecución + medición de tiempo | ✅ Listo | `ejecutar_inferencia()` mide latencia en ms |
| 7. Posprocesamiento | Clase final + confianza (softmax) | ✅ Listo | `ejecutar_inferencia()` aplica softmax y decodifica clase vía `config/clases.txt` |
| 8. Generación de salida | Presentación legible + persistencia opcional | ✅ Listo | `mostrar_resultado()` (consola) y `persistir_resultados()` (JSON/CSV en `outputs/`) |
| 9. Registro y logs | Logging con control de errores | ✅ Listo | Módulo `logging` escribe en `logs/pipeline.log` y consola; manejo explícito de `FileNotFoundError`, `ValueError`, `RuntimeError`, imágenes corruptas |
| 10. Empaquetado | Estructura `src/`, `models/`, `config/`, `outputs/`, `logs/`, README, requirements | ✅ Listo | Ver estructura de carpetas abajo |

**Veredicto:** el proyecto cumple con las 10 fases del pipeline operativo descrito
en el PDF. La única desviación respecto al ejemplo genérico del material es de
framework (PyTorch en vez de TensorFlow/Keras `.h5`, y PIL+torchvision en vez
de `cv2`), justificada porque así fue entrenado el modelo real (`model.pt`
contiene un `state_dict` de PyTorch, no un modelo Keras).

## 2. Estructura de carpetas

```
proyectoVisionComputadora/
├── README.md
├── requirements.txt
├── .gitignore                    # excluye data/, dataset/, entornos virtuales y artefactos generados
├── ejecutar_pipeline_completo.py # rutina orquestadora: dataset -> entrenamiento -> pruebas
├── config/
│   └── clases.txt              # orden de clases: healthy, late_blight
├── models/
│   ├── model.pt                 # pesos entrenados (state_dict ResNet-18)
│   ├── model_info.json          # versión, hiperparámetros, métricas, dataset
│   └── training_metrics_tizon_papa.csv
├── src/
│   └── pipeline_inferencia.py   # pipeline operativo de inferencia (fases 3-9)
├── ejemplos/
│   ├── test.jpg, test1.jpg, test2.jpg, test2.png
├── preparacion_dataset/
│   └── dataset_tizon_papa.py    # script (convertido desde el notebook) que audita,
│                                 # balancea y exporta el dataset a YOLO/COCO/Pascal VOC
├── entrenamiento/
│   └── entrenar_modelo.py       # script (convertido desde el notebook) que entrena
│                                 # ResNet-18 por transfer learning y guarda model.pt
├── pruebas/
│   └── probar_clasificacion.py  # verifica que las imágenes se clasifican correctamente
│                                 # (ejemplos conocidos + accuracy sobre el split de test)
├── data/                          # NO versionado (ver .gitignore)
│   ├── potato_late_blightanotaciones/        # anotaciones/imágenes fuente (crudo, ~96 MB)
│   └── potato_late_blight_dataset_validated/ # dataset auditado, balanceado y exportado (~553 MB)
├── dataset/                       # NO versionado; se genera/regenera por entrenar_modelo.py
├── outputs/                      # resultados persistidos (JSON/CSV), vacío por defecto
└── logs/                          # logs de ejecución (pipeline.log), vacío por defecto
```

> **Nota de tamaño y control de versiones:** `data/` pesa ~650 MB en total
> (dataset crudo + exportado en YOLO/COCO/Pascal VOC). No se copió el ZIP
> original (`anotaciones y dataset potato late.zip`, 639 MB) por ser redundante
> con `potato_late_blightanotaciones/`. El `.gitignore` excluye `data/`,
> `dataset/` (intermedio de entrenamiento), entornos virtuales/caché y los
> artefactos generados en `outputs/`/`logs/`, para no subir datasets ni
> dependencias de instalación al repositorio. Si alguien clona el proyecto sin
> `data/`, debe colocar manualmente `potato_late_blightanotaciones/` (o ejecutar
> su propio proceso de anotación) antes de correr la rutina completa.

## 3. Instalación

Ejecuta los comandos desde la raíz del repositorio.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Uso

Clasificar una sola imagen:
```bash
python src/pipeline_inferencia.py --imagen ejemplos/test.jpg
```

Clasificar todas las imágenes de un directorio y guardar resultados:
```bash
python src/pipeline_inferencia.py --directorio ejemplos/ --salida outputs/resultados.json
```

Forzar CPU/GPU:
```bash
python src/pipeline_inferencia.py --imagen ejemplos/test.jpg --dispositivo cpu
```

## 5. Rutina completa: dataset → modelo → prueba de clasificación

Para regenerar todo el proyecto desde cero (requiere `data/potato_late_blightanotaciones/`
con las anotaciones crudas; no se versiona en git, ver sección 2):

```bash
python ejecutar_pipeline_completo.py
```

Esto ejecuta en orden, deteniéndose si alguno falla:

1. **Generar el dataset** — `preparacion_dataset/dataset_tizon_papa.py`: audita,
   balancea y exporta `data/potato_late_blight_dataset_validated/` (YOLO/COCO/Pascal VOC).
2. **Generar el modelo** — `entrenamiento/entrenar_modelo.py`: reorganiza
   `data/.../yolo/images/{train,val}` al formato `ImageFolder` (`dataset/`),
   entrena ResNet-18 por transfer learning (10 épocas, SGD + CyclicLR) y
   guarda `models/model.pt` + `models/training_metrics_tizon_papa.csv`.
3. **Probar la clasificación** — `pruebas/probar_clasificacion.py`: verifica
   que `ejemplos/test.jpg` se clasifique como `healthy` y `ejemplos/test1.jpg`
   como `late_blight`, y calcula el accuracy sobre las 58 imágenes de la
   partición de test (exige ≥90% por defecto); guarda el reporte en
   `outputs/reporte_clasificacion.json`.

Pasos individuales (también se pueden ejecutar por separado):
```bash
cd data && python ../preparacion_dataset/dataset_tizon_papa.py && cd ..
python entrenamiento/entrenar_modelo.py --epocas 10
python pruebas/probar_clasificacion.py --umbral 0.90
```

Para saltar pasos o ajustar la rutina:
```bash
# Solo reentrenar y volver a probar (el dataset ya existe)
python ejecutar_pipeline_completo.py --saltar-dataset

# Entrenamiento rápido de validación (no usar para el modelo final)
python ejecutar_pipeline_completo.py --epocas 2 --umbral 0.80
```

> **Validado:** se probó `entrenamiento/entrenar_modelo.py` con 2 épocas
> (guardando en archivos temporales, sin tocar `models/model.pt`) y reprodujo
> el mismo `val_accuracy` (99.15% en la época 2) documentado en
> `training_metrics_tizon_papa.csv`. También se ejecutó
> `pruebas/probar_clasificacion.py` contra el modelo de producción: acertó los
> dos ejemplos conocidos y obtuvo 98.28% (57/58) sobre el split de test.

## 6. Reproducibilidad

- El preprocesamiento (`TRANSFORM_INFERENCIA`) replica exactamente el `transform`
  usado en el notebook de entrenamiento: `Resize((224,224))`, `ToTensor()`,
  `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`.
- El orden de clases (`config/clases.txt`: `healthy`=0, `late_blight`=1) coincide
  con el orden alfabético que asigna `torchvision.datasets.ImageFolder` durante
  el entrenamiento.
- `models/model.pt` corresponde a los pesos de la época con mejor `val_accuracy`
  (99.15%), no necesariamente la última época entrenada (ver `model_info.json`).
