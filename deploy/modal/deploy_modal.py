"""
Despliegue del servicio de clasificacion binaria radiologica en Modal.

Modal ejecuta la misma aplicacion FastAPI del proyecto, de modo que la API
REST desplegada es identica a la que se prueba en local con Docker: no hay
una version "de nube" distinta de la version "de desarrollo".

Uso:
    modal setup                                   # una sola vez, autentica
    modal serve deploy/modal/deploy_modal.py      # pruebas, URL temporal
    modal deploy deploy/modal/deploy_modal.py     # despliegue permanente
"""

from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent.parent
SERVICIO = RAIZ / "python/xrays_evaluation"
MODELO = "xrays_evaluation_model_medium_v1.pt"

app = modal.App("clasificacion-binaria-radiologica")

imagen = (
    modal.Image.debian_slim(python_version="3.11")
    # libgl1 no aparece: se usa opencv-python-headless, que no lo necesita.
    .apt_install("libglib2.0-0", "libgomp1")
    # Ruedas de CPU: las de PyPI incluyen CUDA y pesan varios gigabytes,
    # inutiles aqui porque el servicio corre sobre CPU.
    .pip_install(
        "torch==2.9.1+cpu",
        "torchvision==0.24.1+cpu",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "ultralytics==8.3.230",
        "opencv-python-headless==4.12.0.88",
        "fastapi>=0.115,<1.0",
        "pydantic>=2.0,<3.0",
        "numpy==2.2.6",
    )
    # ultralytics declara opencv-python como dependencia, asi que pip lo
    # instala pese a que ya pedimos la variante headless. Ambas comparten el
    # directorio site-packages/cv2, de modo que desinstalar una deja el
    # modulo mutilado (falta hasta cv2.IMREAD_COLOR). Por eso se desinstalan
    # LAS TRES variantes y despues se instala headless desde cero: es la
    # unica secuencia que garantiza un cv2 integro.
    # polars (150 MB) no interviene en inferencia y solo alarga el arranque.
    .run_commands(
        "pip uninstall -y opencv-python opencv-contrib-python "
        "opencv-contrib-python-headless opencv-python-headless || true",
        "pip install --no-cache-dir opencv-python-headless==4.12.0.88",
        "rm -rf /usr/local/lib/python3.11/site-packages/polars "
        "/usr/local/lib/python3.11/site-packages/_polars_runtime_32 || true",
        # Comprobacion en tiempo de construccion: si cv2 quedo incompleto,
        # la imagen falla aqui y no en la primera peticion del usuario.
        "python -c \"import cv2, numpy; "
        "assert hasattr(cv2, 'IMREAD_COLOR'), 'cv2 incompleto'; "
        "cv2.imdecode(numpy.zeros((10,), dtype=numpy.uint8), cv2.IMREAD_COLOR); "
        "print('cv2 verificado:', cv2.__version__)\"",
    )
    .env({
        "PYTHONPATH": "/app",
        "XRAY_MODEL_PATH": f"/app/models/YOLO/{MODELO}",
        # Rutas escribibles: el sistema de archivos del contenedor es de
        # solo lectura salvo el directorio temporal. ultralytics anade el
        # subdirectorio "Ultralytics" por su cuenta, de ahi que se apunte a
        # la raiz temporal y no a una ruta ya terminada en ese nombre.
        "YOLO_CONFIG_DIR": "/tmp",
        "MPLCONFIGDIR": "/tmp/matplotlib",
    })
    .workdir("/app")
    .add_local_dir(SERVICIO / "src", "/app/src")
    .add_local_file(SERVICIO / "models/YOLO" / MODELO, f"/app/models/YOLO/{MODELO}")
)


@app.function(
    image=imagen,
    cpu=1.0,
    # 512 MB provoca cierre por falta de memoria (verificado en local):
    # PyTorch ocupa entre 250 y 300 MB solo con cargarse.
    memory=2048,
    # Segundos de inactividad antes de apagar el contenedor. Un valor bajo
    # ahorra credito; uno alto evita arranques en frio. 60 s equilibra ambos
    # durante una demostracion, donde las peticiones llegan seguidas.
    scaledown_window=60,
    timeout=300,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    from src.server.app import app as aplicacion_fastapi
    return aplicacion_fastapi
