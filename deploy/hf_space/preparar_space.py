"""
Ensambla el contenido que se publica en el Space de Hugging Face.

El Space es un repositorio git independiente del repositorio del proyecto, y
solo debe contener lo necesario para ejecutar la inferencia: el codigo del
servicio, el modelo desplegado y los archivos de construccion. El codigo de
entrenamiento, los otros dos modelos y el benchmark se quedan fuera.

Uso:
    python deploy/hf_space/preparar_space.py [--destino RUTA]
"""

import argparse
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SERVICIO = RAIZ / "python/xrays_evaluation"
MODELO = "xrays_evaluation_model_medium_v1.pt"

# Subcarpetas de src/ que no intervienen en la inferencia.
EXCLUIR_DE_SRC = {"training", "__pycache__"}

ARCHIVOS_PROPIOS = ["Dockerfile", "README.md", "requirements.txt", ".gitattributes"]


def copiar_src(destino: Path):
    origen = SERVICIO / "src"
    for elemento in origen.iterdir():
        if elemento.name in EXCLUIR_DE_SRC:
            continue
        objetivo = destino / "src" / elemento.name
        if elemento.is_dir():
            shutil.copytree(elemento, objetivo,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            objetivo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(elemento, objetivo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=str(AQUI / "build"),
                    help="carpeta donde se ensambla el Space "
                         "(usar el clon del repositorio de Hugging Face)")
    args = ap.parse_args()

    destino = Path(args.destino).resolve()
    destino.mkdir(parents=True, exist_ok=True)

    # Se limpia solo lo que este script genera, para no borrar el .git del
    # clon del Space si el destino es ese repositorio.
    for sub in ("src", "models"):
        if (destino / sub).exists():
            shutil.rmtree(destino / sub)

    print(f"Ensamblando Space en: {destino}\n")

    copiar_src(destino)
    print("  src/                    copiado (sin training/)")

    origen_modelo = SERVICIO / "models/YOLO" / MODELO
    if not origen_modelo.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {origen_modelo}")
    destino_modelo = destino / "models/YOLO" / MODELO
    destino_modelo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen_modelo, destino_modelo)
    print(f"  models/YOLO/{MODELO}  ({origen_modelo.stat().st_size / 1024**2:.1f} MB)")

    for nombre in ARCHIVOS_PROPIOS:
        origen = AQUI / nombre
        if origen.exists():
            shutil.copy2(origen, destino / nombre)
            print(f"  {nombre}")

    total = sum(f.stat().st_size for f in destino.rglob("*")
                if f.is_file() and ".git" not in f.parts)
    print(f"\nTotal ensamblado: {total / 1024**2:.1f} MB")
    print(f"\nSiguiente paso:\n  cd {destino}\n  git add -A && git commit -m \"Actualiza servicio\"\n  git push")


if __name__ == "__main__":
    main()
