# Sistema de clasificación binaria radiológica basado en algoritmos de Deep Learning

Trabajo de grado de Ingeniería en Sistemas. Servicio de apoyo al diagnóstico que
clasifica radiografías de tórax en **Normal** o **Anomaly** mediante una red neuronal
convolucional, expone el resultado como API REST y explica su decisión con mapas de
atención Grad-CAM.

> **Aviso.** Herramienta de carácter académico. No constituye un dispositivo médico ni
> sustituye el juicio de un profesional de la salud. Sus resultados no deben emplearse
> para tomar decisiones clínicas.

---

## Arquitectura

```text
Navegador  ──►  Frontend (Next.js, Vercel)  ──►  API (FastAPI, Modal)  ──►  YOLO11m-cls
                        │                                │
                  intermediario                    Grad-CAM + filtro
                  y limitador                      de plausibilidad
```

El navegador nunca contacta con el servicio de inferencia: las rutas de
`frontend/app/api/` actúan como intermediario, de modo que la URL del despliegue vive
solo en el servidor. Así el endpoint no es extraíble desde el cliente y las peticiones
quedan en el mismo origen.

| Componente | Ubicación | Descripción |
|---|---|---|
| Servicio de inferencia | [`python/xrays_evaluation/`](python/xrays_evaluation) | FastAPI, YOLO11m-cls, Grad-CAM |
| Despliegue | [`deploy/modal/`](deploy/modal) | Definición para Modal |
| Interfaz web | [`frontend/`](frontend) | Next.js, visor clínico |
| Utilidades | [`utilidades/`](utilidades) | Pruebas de aceptación y herramientas |

---

## Resultados

Evaluación sobre **900 radiografías reservadas**, ajenas al entrenamiento, verificadas
mediante hash exacto y perceptual calibrado.

| Modelo | Exactitud | Sensibilidad | Especificidad | AUC | Latencia | Tamaño |
|---|---|---|---|---|---|---|
| YOLO11n-cls | 0.9656 | 0.9786 | 0.9541 | 0.9962 | 6.5 ms | 3 MB |
| **YOLO11m-cls** | **0.9889** | **0.9810** | **0.9958** | **0.9998** | **22.6 ms** | **20 MB** |
| YOLO11x-cls | 0.9900 | 0.9834 | 0.9958 | 0.9991 | 55.7 ms | 54 MB |

Se desplegó la variante *medium*: la diferencia con *xlarge* es de **una imagen entre
900**, dentro del ruido estadístico, a cambio de 2,7 veces más tamaño y latencia.

### Calibración

El modelo no está sobreajustado ni sobreconfiado. Su confianza media es de 0.9943
cuando acierta y **0.7077 cuando se equivoca**, y ninguno de sus 10 errores superó una
confianza de 0.99. El Error de Calibración Esperado es de **0.0030**.

De ahí se deriva el criterio de derivación que implementa el sistema: un umbral de
**0.99** habría capturado el 100% de los errores revisando solo el 5,8% de los casos.

---

## Aportes sobre el proyecto base

Este trabajo parte de [AprendeIngenia/deep_learning_services](https://github.com/AprendeIngenia/deep_learning_services),
material educativo de inGeniia.co. Las contribuciones propias son:

**Grad-CAM real.** El proyecto base promediaba los canales de activación sin calcular
gradientes, de modo que el mapa era idéntico para ambas clases. Se implementó Grad-CAM
según Selvaraju et al. (2017), retropropagando el logit de la clase predicha. El mapa
pasó a ser específico de la clase, con correlación negativa entre el de cada clase
sobre la misma imagen. Al corregir además el preprocesamiento, la latencia de esta
etapa bajó de 507–1353 ms a ~83 ms constantes.

**Auditoría de fuga de datos.** La verificación por hash MD5 resulta insuficiente
porque el reprocesamiento de las imágenes altera los bytes sin alterar el contenido. Se
implementó una verificación perceptual calibrada contra la distribución nula del propio
conjunto. Reveló que el conjunto denominado *benchmarking* por el proyecto base
comparte el 73% de sus imágenes con el entrenamiento, por lo que **no constituye
validación externa** pese a su denominación.

**Filtro de plausibilidad.** Un clasificador binario no puede responder "esto no es una
radiografía". Se midió: el ruido en color obtenía *Anomaly* con 99,90% de confianza. Se
añadió una comprobación previa cuyos umbrales se derivaron midiendo 1.100 radiografías
reales.

**Optimización del empaquetado.** La imagen pasó de 2,56 GB a 1,38 GB al detectar que
OpenCV estaba instalado por duplicado, lo que arrastraba 41 paquetes de controladores
gráficos innecesarios en un servidor sin pantalla.

---

## Puesta en marcha

### Servicio de inferencia

```bash
docker build -t xrays-service -f deploy/hf_space/build/Dockerfile deploy/hf_space/build
```

```bash
docker run -d -p 8080:8080 --name xrays-service xrays-service
```

Documentación interactiva en `http://localhost:8080/docs`.

### Interfaz web

```bash
cd frontend && npm install && npm run dev
```

Requiere un archivo `.env.local` con la variable documentada en
[`frontend/.env.example`](frontend/.env.example).

### Pruebas de aceptación

```bash
python utilidades/verificar_sistema.py --url http://localhost:8080
```

Verifica clasificación, contrato de respuesta, validez del Grad-CAM, códigos de error,
rechazo de entradas no radiográficas y política CORS.

---

## Reproducir la evaluación

Los scripts requieren el conjunto de datos, cuya ubicación se indica con la variable de
entorno `XRAY_DATASET_DIR`.

| Script | Qué produce |
|---|---|
| [`benchmark_tesis.py`](python/xrays_evaluation/tests/benchmark_tesis.py) | Métricas y figuras comparativas de los tres modelos |
| [`analisis_calibracion.py`](python/xrays_evaluation/tests/analisis_calibracion.py) | Distribución de confianza y diagrama de fiabilidad |
| [`verificar_fuga_perceptual.py`](python/xrays_evaluation/tests/verificar_fuga_perceptual.py) | Auditoría de solapamiento entre particiones |
| [`calibrar_plausibilidad.py`](python/xrays_evaluation/tests/calibrar_plausibilidad.py) | Umbrales del filtro de entrada |

---

## Limitaciones

- El conjunto de datos corresponde a **radiografías pediátricas de tórax**. El
  rendimiento no está validado en población adulta ni en otras patologías.
- No se dispone de validación sobre una población independiente: el conjunto que el
  proyecto base ofrece como externo resultó solapado con el entrenamiento.
- El filtro de plausibilidad descarta entradas manifiestamente ajenas al dominio, pero
  no detecta una radiografía de otra región anatómica.
- En el nivel gratuito de la plataforma el servicio se detiene tras un minuto sin
  tráfico; la primera petición posterior tarda unos 11 segundos frente a los 2
  habituales.

---

## Licencia

AGPL-3.0, heredada de [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

Este software implementa arquitecturas de Deep Learning basadas en los materiales
educativos de inGeniia.co.
