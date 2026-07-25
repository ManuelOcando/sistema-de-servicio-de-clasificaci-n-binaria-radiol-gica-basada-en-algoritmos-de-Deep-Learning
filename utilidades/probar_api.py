"""
Prueba el servicio de clasificacion radiologica, este desplegado o en local.

Envia una radiografia, muestra la prediction y guarda el mapa de calor
recibido, de modo que sirve tanto para verificar un despliegue como para
generar evidencias graficas.

La URL del servicio se indica con --url o con la variable de entorno
XRAY_API_URL. Por defecto apunta al contenedor local.

Ejemplos:
    # Contra el contenedor local
    python utilidades/probar_api.py --imagen ruta/a/rx.jpg

    # Contra un despliegue en la nube
    python utilidades/probar_api.py --imagen ruta/a/rx.jpg --url https://TU-DESPLIEGUE

    # Midiendo arranque en frio frente a peticiones en caliente
    python utilidades/probar_api.py --imagen ruta/a/rx.jpg --repeticiones 3
"""

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# La URL del despliegue no se escribe en el codigo: este repositorio es
# publico y el servicio se factura por uso, de modo que difundir el endpoint
# permitiria a terceros consumir el credito de la cuenta. Se toma de la
# variable de entorno XRAY_API_URL o del parametro --url.
URL_POR_DEFECTO = os.getenv("XRAY_API_URL", "http://127.0.0.1:8080")


def pedir(url: str, ruta_imagen: Path, timeout: int) -> tuple:
    b64 = base64.b64encode(ruta_imagen.read_bytes()).decode("utf-8")
    cuerpo = json.dumps({"image_base64": b64}).encode("utf-8")
    peticion = urllib.request.Request(
        f"{url}/cnn_xray_demo", data=cuerpo,
        headers={"Content-Type": "application/json"},
    )
    inicio = time.time()
    with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
        datos = json.loads(respuesta.read())
    return datos, time.time() - inicio


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagen", required=True, help="radiografia a clasificar")
    ap.add_argument("--url", default=URL_POR_DEFECTO, help="URL base del servicio")
    ap.add_argument("--repeticiones", type=int, default=1,
                    help="numero de peticiones; la primera incluye el arranque en frio")
    # Por defecto se escribe junto a la imagen de entrada y no en el directorio
    # actual: PowerShell arranca en C:\Windows\System32, que no es escribible
    # sin permisos de administrador.
    ap.add_argument("--salida", default=None,
                    help="prefijo de los archivos generados "
                         "(por defecto, junto a la imagen de entrada)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    imagen = Path(args.imagen)
    if not imagen.is_file():
        raise SystemExit(f"No existe la imagen: {imagen}")

    url = args.url.rstrip("/")
    print(f"Servicio : {url}")
    print(f"Imagen   : {imagen.name} ({imagen.stat().st_size / 1024:.0f} KB)\n")

    print("Comprobando estado...")
    try:
        inicio = time.time()
        with urllib.request.urlopen(f"{url}/health", timeout=args.timeout) as r:
            print(f"  {r.read().decode()}  ({time.time() - inicio:.1f} s)\n")
    except urllib.error.URLError as e:
        raise SystemExit(f"El servicio no responde: {e}")

    tiempos = []
    for n in range(1, args.repeticiones + 1):
        try:
            datos, transcurrido = pedir(url, imagen, args.timeout)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Error HTTP {e.code}: {e.read().decode()[:400]}")

        tiempos.append(transcurrido)
        p, perf = datos["prediction"], datos["performance"]
        etiqueta = "(arranque en frio)" if n == 1 and args.repeticiones > 1 else ""
        print(f"Peticion {n}/{args.repeticiones} {etiqueta}")
        print(f"  Prediccion   : {p['label']}   confianza {p['confidence']:.4f}")
        print(f"  Inferencia   : {perf['inference_time_ms']:.1f} ms")
        print(f"  Grad-CAM     : {perf['explainability_time_ms']:.1f} ms")
        print(f"  Ida y vuelta : {transcurrido:.2f} s\n")

    # Se guardan las imagenes de la ultima respuesta
    prefijo = Path(args.salida) if args.salida else imagen.parent / f"resultado_{imagen.stem}"
    for clave, sufijo in (("heatmap_base64", "heatmap"), ("overlay_base64", "overlay")):
        s = datos["explainability"][clave]
        destino = prefijo.with_name(f"{prefijo.name}_{sufijo}.jpg")
        try:
            destino.write_bytes(base64.b64decode(s.split(",")[1] if "," in s else s))
            print(f"Guardado: {destino}")
        except OSError as e:
            print(f"No se pudo escribir en {destino}: {e}")
            print("  Indica una ruta escribible con --salida, por ejemplo:")
            print(f"  --salida {Path.home() / 'Downloads' / 'resultado'}")
            break

    if len(tiempos) > 1:
        print(f"\nPrimera peticion: {tiempos[0]:.2f} s")
        print(f"Resto (media)   : {sum(tiempos[1:]) / len(tiempos[1:]):.2f} s")


if __name__ == "__main__":
    main()
