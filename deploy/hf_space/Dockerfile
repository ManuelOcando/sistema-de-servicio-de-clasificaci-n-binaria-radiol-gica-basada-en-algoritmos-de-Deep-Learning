# Imagen del servicio de clasificacion binaria radiologica para Hugging Face Spaces.
#
# Diferencias respecto a la imagen original del proyecto base:
#   - Elimina la instalacion duplicada de OpenCV (opencv-python +
#     opencv-python-headless). Al quedarse solo con la variante headless
#     deja de ser necesaria libgl1, que arrastraba 41 paquetes del sistema
#     (mesa, LLVM, drivers graficos) inutiles en un servidor sin pantalla.
#   - Empaqueta unicamente el modelo medium, que es el desplegado segun el
#     benchmark comparativo.
#   - Se ejecuta como usuario sin privilegios (UID 1000), como requiere
#     Hugging Face Spaces.
#   - Escucha en el puerto 7860, el que expone Spaces por defecto.

# --- Etapa 1: construccion ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Ultralytics declara opencv-python como dependencia, de modo que pip lo
# instala aunque nosotros pidamos la variante headless: pip los trata como
# distribuciones distintas pese a que ambas proveen el modulo cv2. Se
# desinstala y se reinstala headless al final para dejar una sola copia.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.12.0.88

# Poda de artefactos que no intervienen en la inferencia.
#
# Solo se eliminan rutas que ningun modulo importa al cargarse. En concreto
# NO se toca torchvision/datasets: aunque el servicio no usa datasets,
# torchvision/__init__.py lo importa al inicializarse y borrarlo deja el
# paquete inservible ("cannot import name 'datasets'"). Lo mismo aplica a
# los submodulos de torch/utils.
#
# polars (150 MB) llega como dependencia de ultralytics pero solo interviene
# en utilidades de manejo de datos durante el entrenamiento; se verifico que
# la inferencia funciona sin el.
RUN SP=/usr/local/lib/python3.11/site-packages && \
    rm -rf $SP/torch/include $SP/torch/test \
           $SP/polars $SP/_polars_runtime_32 \
           $SP/pip $SP/setuptools $SP/pkg_resources 2>/dev/null || true && \
    find $SP -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find $SP -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find $SP -type d -name "test" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find $SP -name "*.pyc" -delete 2>/dev/null || true && \
    find $SP -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# --- Etapa 2: ejecucion ---
FROM python:3.11-slim AS runtime

# libgl1 ya no aparece: opencv-python-headless no lo necesita.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Spaces ejecuta el contenedor como usuario sin privilegios.
RUN useradd -m -u 1000 usuario

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    HOME=/home/usuario \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

WORKDIR /app
COPY --chown=usuario:usuario src/ ./src/
COPY --chown=usuario:usuario models/ ./models/

USER usuario

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
