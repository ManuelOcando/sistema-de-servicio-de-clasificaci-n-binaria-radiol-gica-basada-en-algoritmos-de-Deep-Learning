"""
Pruebas de aceptacion del sistema de clasificacion binaria radiologica.

Verifica de forma automatica cada afirmacion que el sistema sostiene: que
clasifica correctamente ambas clases, que la respuesta cumple el contrato
declarado, que el mapa de calor es valido y especifico de la clase, que los
errores se manejan con los codigos HTTP adecuados y que la politica CORS
esta activa.

Cada comprobacion informa PASA o FALLA y el proceso termina con codigo
distinto de cero si alguna falla, de modo que sirve como prueba de humo
antes de fijar un despliegue definitivo.

La URL se indica con --url o con la variable de entorno XRAY_API_URL.

Uso:
    python utilidades/verificar_sistema.py --url https://TU-DESPLIEGUE
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

CAMPOS = {
    "prediction": {"label": str, "confidence": float, "class_id": int},
    "explainability": {"heatmap_base64": str, "overlay_base64": str, "description": str},
    "performance": {"preprocess_time_ms": float, "inference_time_ms": float,
                    "explainability_time_ms": float, "total_latency_ms": float,
                    "model_used": str},
}

resultados = []


def comprobar(nombre: str, condicion: bool, detalle: str = ""):
    estado = "PASA " if condicion else "FALLA"
    print(f"  [{estado}] {nombre}" + (f"  -> {detalle}" if detalle else ""))
    resultados.append((nombre, condicion))
    return condicion


def peticion(url: str, ruta: str, cuerpo=None, cabeceras=None, timeout=300):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    cab = {"Content-Type": "application/json"}
    cab.update(cabeceras or {})
    req = urllib.request.Request(f"{url}{ruta}", data=datos, headers=cab)
    inicio = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers), time.time() - inicio
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers), time.time() - inicio


def b64_de(ruta: Path) -> str:
    return base64.b64encode(ruta.read_bytes()).decode("utf-8")


def es_jpeg(datos: bytes) -> bool:
    return datos[:2] == b"\xff\xd8" and datos[-2:] == b"\xff\xd9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("XRAY_API_URL", "http://127.0.0.1:8080"))
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"Sistema bajo prueba: {url}\n")

    # --- 1. Disponibilidad ---
    print("1. Disponibilidad del servicio")
    estado, cuerpo, _, t = peticion(url, "/health")
    comprobar("El servicio responde", estado == 200, f"HTTP {estado} en {t:.1f} s")
    if estado != 200:
        print("\nEl servicio no responde; se abandonan las comprobaciones.")
        return 1
    salud = json.loads(cuerpo)
    comprobar("Declara estado correcto", salud.get("status") == "ok", str(salud))
    comprobar("Declara el modelo desplegado", "model" in salud, salud.get("model", ""))

    # --- 2. Clasificacion ---
    print("\n2. Clasificacion de ambas clases")
    respuestas = {}
    for archivo, esperada in (("anomaly_rx_test.jpeg", "Anomaly"),
                              ("normal_rx_test.jpeg", "Normal")):
        ruta = EJEMPLOS / archivo
        if not ruta.exists():
            comprobar(f"Imagen de ejemplo {archivo}", False, "no encontrada")
            continue
        estado, cuerpo, _, t = peticion(url, "/cnn_xray_demo",
                                        {"image_base64": b64_de(ruta)})
        if not comprobar(f"{archivo} responde", estado == 200, f"HTTP {estado}"):
            continue
        datos = json.loads(cuerpo)
        respuestas[esperada] = datos
        obtenida = datos["prediction"]["label"]
        comprobar(f"{archivo} clasificada como {esperada}", obtenida == esperada,
                  f"obtenido {obtenida}, confianza "
                  f"{datos['prediction']['confidence']:.4f}, {t:.2f} s")

    if len(respuestas) < 2:
        print("\nNo se pudieron obtener ambas respuestas.")
        return 1

    # --- 3. Contrato de la respuesta ---
    print("\n3. Contrato de la respuesta")
    muestra = respuestas["Anomaly"]
    for seccion, campos in CAMPOS.items():
        if not comprobar(f"Contiene la seccion '{seccion}'", seccion in muestra):
            continue
        faltantes = [c for c in campos if c not in muestra[seccion]]
        comprobar(f"'{seccion}' tiene todos sus campos", not faltantes,
                  f"faltan {faltantes}" if faltantes else "completa")
        malos = [c for c, tipo in campos.items()
                 if c in muestra[seccion]
                 and not isinstance(muestra[seccion][c], (tipo, int) if tipo is float else tipo)]
        comprobar(f"'{seccion}' respeta los tipos declarados", not malos,
                  f"incorrectos {malos}" if malos else "correctos")

    conf = muestra["prediction"]["confidence"]
    comprobar("La confianza esta en el rango [0, 1]", 0.0 <= conf <= 1.0, f"{conf}")

    # --- 4. Explicabilidad ---
    print("\n4. Explicabilidad (Grad-CAM)")
    mapas = {}
    for etiqueta, datos in respuestas.items():
        for clave, sufijo in (("heatmap_base64", "mapa"), ("overlay_base64", "superposicion")):
            s = datos["explainability"][clave]
            try:
                crudo = base64.b64decode(s.split(",")[1] if "," in s else s)
            except Exception as e:
                comprobar(f"{etiqueta}: {sufijo} decodificable", False, str(e))
                continue
            comprobar(f"{etiqueta}: {sufijo} es un JPEG valido", es_jpeg(crudo),
                      f"{len(crudo) / 1024:.0f} KB")
            if clave == "heatmap_base64":
                mapas[etiqueta] = crudo

    if len(mapas) == 2:
        distintos = mapas["Anomaly"] != mapas["Normal"]
        comprobar("El mapa difiere entre ambas radiografias", distintos,
                  "especifico de la entrada" if distintos else "identico: sospechoso")

    # --- 5. Manejo de errores ---
    print("\n5. Manejo de errores")
    estado, _, _, _ = peticion(url, "/cnn_xray_demo", {"image_base64": "esto-no-es-base64"})
    comprobar("Base64 invalido devuelve 400", estado == 400, f"HTTP {estado}")

    estado, _, _, _ = peticion(url, "/cnn_xray_demo", {})
    comprobar("Peticion sin el campo requerido devuelve 422", estado == 422, f"HTTP {estado}")

    estado, _, _, _ = peticion(url, "/cnn_xray_demo",
                               {"image_base64": base64.b64encode(b"no soy una imagen").decode()})
    comprobar("Contenido que no es imagen devuelve 400", estado == 400, f"HTTP {estado}")

    # --- 6. Rechazo de entradas que no son radiografias ---
    print("\n6. Rechazo de entradas no radiográficas")
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [OMITIDA] requiere numpy y opencv para generar los casos")
    else:
        rng = np.random.default_rng(7)
        degradado = np.zeros((600, 800, 3), dtype=np.uint8)
        degradado[:, :, 0] = np.linspace(200, 40, 800)[None, :]
        degradado[:, :, 1] = np.linspace(120, 190, 600)[:, None]
        degradado[:, :, 2] = 60

        invalidas = {
            "ruido a color": rng.integers(0, 255, (600, 600, 3), dtype=np.uint8),
            "blanco uniforme": np.full((600, 600, 3), 255, dtype=np.uint8),
            "degradado de color": degradado,
        }

        for nombre, arreglo in invalidas.items():
            _, buffer = cv2.imencode(".jpg", arreglo)
            b64 = base64.b64encode(buffer.tobytes()).decode()
            estado, cuerpo, _, _ = peticion(url, "/cnn_xray_demo", {"image_base64": b64})
            detalle = ""
            if estado == 200:
                p = json.loads(cuerpo)["prediction"]
                detalle = f"CLASIFICADA como {p['label']} al {p['confidence'] * 100:.1f}%"
            comprobar(f"'{nombre}' se rechaza con 400", estado == 400,
                      detalle or f"HTTP {estado}")

    # --- 7. Politica CORS ---
    print("\n7. Politica CORS")
    estado, _, cabeceras, _ = peticion(url, "/health",
                                       cabeceras={"Origin": "http://localhost:3000"})
    permitido = cabeceras.get("access-control-allow-origin") or \
        cabeceras.get("Access-Control-Allow-Origin")
    comprobar("Autoriza el origen de desarrollo", permitido is not None,
              permitido or "sin cabecera CORS")

    estado, _, cabeceras, _ = peticion(url, "/health",
                                       cabeceras={"Origin": "https://sitio-no-autorizado.example"})
    negado = not (cabeceras.get("access-control-allow-origin") or
                  cabeceras.get("Access-Control-Allow-Origin"))
    comprobar("Rechaza un origen no autorizado", negado,
              "sin cabecera, correcto" if negado else "lo autoriza: revisar")

    # --- Resumen ---
    fallos = [n for n, ok in resultados if not ok]
    print("\n" + "=" * 62)
    print(f"  {len(resultados) - len(fallos)} de {len(resultados)} comprobaciones superadas")
    if fallos:
        print("\n  Fallaron:")
        for n in fallos:
            print(f"    - {n}")
    else:
        print("  Sistema verificado: listo para fijar el despliegue.")
    print("=" * 62)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
