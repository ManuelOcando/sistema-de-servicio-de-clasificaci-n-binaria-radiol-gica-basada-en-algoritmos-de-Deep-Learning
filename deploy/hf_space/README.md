---
title: Clasificacion Binaria Radiologica
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# Sistema de clasificación binaria radiológica

Servicio de clasificación de radiografías de tórax (**Normal** / **Anomaly**) mediante
una red neuronal convolucional, con explicabilidad visual por Grad-CAM.

Desarrollado como trabajo de grado de Ingeniería en Sistemas, tomando como base el
proyecto educativo [deep_learning_services](https://github.com/AprendeIngenia/deep_learning_services)
de inGeniia.co.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/docs` | Documentación interactiva (Swagger) |
| `GET` | `/health` | Verificación de estado del servicio |
| `POST` | `/cnn_xray_demo` | Clasifica una radiografía y genera el mapa de calor |

### Ejemplo de petición

```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
}
```

### Respuesta

```json
{
  "prediction": { "label": "Anomaly", "confidence": 0.9853, "class_id": 0 },
  "explainability": {
    "heatmap_base64": "...",
    "overlay_base64": "...",
    "description": "Red indicates high attention regions."
  },
  "performance": {
    "preprocess_time_ms": 12.5,
    "inference_time_ms": 39.1,
    "explainability_time_ms": 83.3,
    "total_latency_ms": 134.9,
    "model_used": "YOLO11m-cls"
  }
}
```

## Modelo desplegado

`YOLO11m-cls`, seleccionado tras evaluar comparativamente tres variantes sobre dos
conjuntos independientes del entrenamiento:

| Modelo | Exactitud | Sensibilidad | Especificidad | AUC | Latencia | Tamaño |
|---|---|---|---|---|---|---|
| YOLO11n-cls | 0.9850 | 0.9800 | 0.9900 | 0.9998 | 23.9 ms | 3 MB |
| **YOLO11m-cls** | **0.9900** | **0.9800** | **1.0000** | **0.9999** | **41.6 ms** | **20 MB** |
| YOLO11x-cls | 0.9900 | 0.9800 | 1.0000 | 0.9997 | 74.4 ms | 54 MB |

*(Validación externa, 200 imágenes. La variante xlarge no aporta exactitud medible
sobre la medium pese a ser 2.7 veces más grande y lenta.)*

## Explicabilidad

El mapa de calor se genera con **Grad-CAM** (Selvaraju et al., 2017): se retropropaga
el logit de la clase predicha y se ponderan los mapas de activación de la última capa
convolucional por sus gradientes. El resultado es específico de la clase — el mapa de
"Anomaly" difiere del de "Normal" sobre la misma radiografía.

## Aviso

Herramienta de carácter académico. **No constituye un dispositivo médico ni sustituye
el criterio de un profesional de la salud.**

## Licencia

AGPL-3.0, heredada de [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
