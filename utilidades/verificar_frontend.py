"""
Pruebas de aceptación de la interfaz web desplegada.

Complementa a verificar_sistema.py, que prueba el servicio de inferencia. Aquí
se verifica la capa que ve el usuario: que la página sirva, que el
intermediario clasifique correctamente, que propague los errores del servicio
y, sobre todo, que la URL del servicio de inferencia NO llegue al navegador.

Esa última comprobación es la que sostiene el modelo de protección: el
servicio se factura por uso, de modo que si su dirección apareciera en el
código del cliente, cualquiera podría consumir el saldo de la cuenta.

Uso:
    python utilidades/verificar_frontend.py --url https://TU-DESPLIEGUE
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EJEMPLOS = RAIZ / "python/xrays_evaluation/examples/images"

resultados = []


def comprobar(nombre: str, condicion: bool, detalle: str = "") -> bool:
    print(f"  [{'PASA ' if condicion else 'FALLA'}] {nombre}" + (f"  -> {detalle}" if detalle else ""))
    resultados.append((nombre, condicion))
    return condicion


def peticion(url: str, ruta: str, cuerpo=None, timeout: int = 120, crudo: bytes = None):
    datos = crudo if crudo is not None else (
        json.dumps(cuerpo).encode() if cuerpo is not None else None
    )
    req = urllib.request.Request(
        f"{url}{ruta}", data=datos, headers={"Content-Type": "application/json"}
    )
    inicio = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers), time.time() - inicio
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers), time.time() - inicio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("XRAY_FRONTEND_URL", "http://127.0.0.1:3000"))
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"Interfaz bajo prueba: {url}\n")

    # --- 1. La página se sirve ---
    print("1. Disponibilidad de la interfaz")
    estado, cuerpo, _, t = peticion(url, "/")
    if not comprobar("La página responde", estado == 200, f"HTTP {estado} en {t:.1f} s"):
        return 1
    html = cuerpo.decode("utf-8", errors="ignore")
    comprobar("Sirve la aplicación esperada",
              "Clasificación binaria radiológica" in html)
    comprobar("Incluye el aviso de uso académico",
              "no constituye un dispositivo médico" in html.lower())

    # --- 2. El endpoint no se filtra ---
    print("\n2. Protección del servicio de inferencia")
    filtrado = "modal.run" in html
    comprobar("La URL del servicio NO aparece en el HTML", not filtrado,
              "expuesta: revisar la configuración" if filtrado
              else f"{len(html)} caracteres analizados")

    # --- 3. Clasificación a través del intermediario ---
    print("\n3. Clasificación")
    estado, cuerpo, _, t = peticion(url, "/api/despertar", timeout=90)
    comprobar("La ruta de despertado responde", estado == 200, f"{cuerpo.decode()[:40]} ({t:.1f} s)")

    for archivo, esperada in (("anomaly_rx_test.jpeg", "Anomaly"),
                              ("normal_rx_test.jpeg", "Normal")):
        ruta = EJEMPLOS / archivo
        if not ruta.exists():
            comprobar(f"Imagen de ejemplo {archivo}", False, "no encontrada")
            continue
        b64 = base64.b64encode(ruta.read_bytes()).decode()
        estado, cuerpo, _, t = peticion(url, "/api/clasificar", {"image_base64": b64})
        if not comprobar(f"{archivo}: HTTP 200", estado == 200, f"HTTP {estado}"):
            print(f"          {cuerpo.decode()[:250]}")
            continue
        datos = json.loads(cuerpo)
        comprobar(f"{archivo}: clasificada como {esperada}",
                  datos["prediction"]["label"] == esperada,
                  f"{datos['prediction']['label']} al "
                  f"{datos['prediction']['confidence'] * 100:.2f}% en {t:.2f} s")
        comprobar(f"{archivo}: incluye el mapa de atención",
                  len(datos["explainability"]["overlay_base64"]) > 1000)

    # --- 4. Propagación de errores ---
    print("\n4. Propagación de errores del servicio")
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [OMITIDA] el caso no radiográfico requiere numpy y opencv")
    else:
        rng = np.random.default_rng(7)
        _, buffer = cv2.imencode(".jpg", rng.integers(0, 255, (500, 500, 3), dtype=np.uint8))
        b64 = base64.b64encode(buffer.tobytes()).decode()
        estado, cuerpo, _, _ = peticion(url, "/api/clasificar", {"image_base64": b64})
        detalle = ""
        if estado == 400:
            detalle = json.loads(cuerpo).get("detail", "")[:60] + "…"
        comprobar("Entrada no radiográfica se rechaza con 400", estado == 400,
                  detalle or f"HTTP {estado}")

    estado, _, _, _ = peticion(url, "/api/clasificar", crudo=b"{ esto no es json")
    comprobar("Cuerpo malformado devuelve 400", estado == 400, f"HTTP {estado}")

    # --- 5. Limitador de peticiones ---
    # Se agota con cuerpos malformados: el limitador los cuenta, pero no
    # llegan al servicio de inferencia, de modo que no consumen saldo.
    print("\n5. Limitador de peticiones")
    print("  (agotando el límite con peticiones que no llegan al servicio)")
    bloqueado, intentos = False, 0
    for intentos in range(1, 31):
        estado, _, cabeceras, _ = peticion(url, "/api/clasificar", crudo=b"{ malformado")
        if estado == 429:
            espera = cabeceras.get("retry-after") or cabeceras.get("Retry-After")
            comprobar("Bloquea el exceso de peticiones", True,
                      f"429 en la petición {intentos}, retry-after={espera}")
            comprobar("Indica el tiempo de espera", espera is not None, str(espera))
            bloqueado = True
            break
    if not bloqueado:
        comprobar("Bloquea el exceso de peticiones", False,
                  f"sin bloqueo tras {intentos} peticiones; en serverless el "
                  f"límite se reparte entre instancias")

    fallos = [n for n, ok in resultados if not ok]
    print("\n" + "=" * 64)
    print(f"  {len(resultados) - len(fallos)} de {len(resultados)} comprobaciones superadas")
    if fallos:
        print("\n  Fallaron:")
        for n in fallos:
            print(f"    - {n}")
    else:
        print("  Interfaz verificada.")
    print("=" * 64)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
