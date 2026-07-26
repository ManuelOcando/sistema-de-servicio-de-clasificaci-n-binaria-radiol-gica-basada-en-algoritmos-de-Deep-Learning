"""
Filtro de plausibilidad radiologica previo a la clasificacion.

Un clasificador binario entrenado unicamente con radiografias carece de la
posibilidad de responder "esto no es una radiografia": la funcion softmax
reparte siempre la totalidad de la probabilidad entre sus dos clases. Medido
sobre el servicio desplegado, una imagen de ruido en color obtiene "Anomaly"
con 99,90% de confianza y un lienzo blanco, 99,99%. El umbral de derivacion a
revision humana tampoco protege de esto, porque esas confianzas quedan por
encima de el.

La comprobacion, por tanto, no puede hacerla el modelo: corresponde a una
etapa anterior. Aqui se verifica que la entrada presente las propiedades
fisicas de una radiografia antes de entregarsela.

Los umbrales no son arbitrarios: se derivaron midiendo las 1.100 radiografias
de los conjuntos de evaluacion (tests/calibrar_plausibilidad.py) y dejando
margen sobre los valores extremos observados. Con ellos, las seis entradas
invalidas de la bateria de prueba se rechazan y ninguna radiografia legitima
resulta afectada.
"""

import cv2
import numpy as np

# --- Umbrales derivados de la medicion sobre radiografias reales ---

# Una radiografia es una imagen en escala de grises: sus tres canales
# coinciden y la saturacion resulta nula. En las 1.100 medidas el valor fue
# exactamente 0,00 sin excepcion. El umbral admite holgura para tolerar
# artefactos de compresion o anotaciones en color de algunos equipos.
SATURACION_MAXIMA = 12.0

# Descarta imagenes de tono uniforme, sin estructura anatomica alguna. El
# minimo observado en radiografias reales fue 26,92.
DESVIACION_MINIMA = 13.46

# Riqueza tonal: una radiografia ocupa buena parte del rango disponible. El
# minimo observado fue 6,17 sobre un maximo teorico de 8.
ENTROPIA_MINIMA = 4.32


class ImagenNoRadiografica(ValueError):
    """La entrada no presenta las propiedades de una radiografia."""


def medir(imagen: np.ndarray) -> dict:
    """Propiedades de plausibilidad de una imagen en BGR."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    saturacion = float(hsv[:, :, 1].mean())

    desviacion = float(gris.std())

    histograma = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    probabilidades = histograma / max(histograma.sum(), 1)
    no_nulas = probabilidades[probabilidades > 0]
    entropia = float(-(no_nulas * np.log2(no_nulas)).sum()) if no_nulas.size else 0.0

    return {
        "saturacion": round(saturacion, 3),
        "desviacion": round(desviacion, 3),
        "entropia": round(entropia, 3),
    }


def revisar(imagen: np.ndarray) -> dict:
    """
    Comprueba que la imagen pueda ser una radiografia.

    :raises ImagenNoRadiografica: si incumple alguna propiedad.
    :return: las metricas medidas, para poder registrarlas.
    """
    m = medir(imagen)
    motivos = []

    if m["saturacion"] > SATURACION_MAXIMA:
        motivos.append(
            "presenta color; una radiografía es una imagen en escala de grises"
        )
    if m["desviacion"] < DESVIACION_MINIMA:
        motivos.append("carece de estructura: su tono es prácticamente uniforme")
    if m["entropia"] < ENTROPIA_MINIMA:
        motivos.append("su rango tonal es demasiado pobre")

    if motivos:
        raise ImagenNoRadiografica(
            "La imagen no parece una radiografía de tórax: "
            + "; ".join(motivos)
            + ". El sistema solo admite radiografías, ya que clasificar otro tipo "
            "de imagen produciría un resultado carente de sentido pese a mostrar "
            "una confianza elevada."
        )

    return m
