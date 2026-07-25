"""
Compatibilidad entre ultralytics y las compilaciones headless de OpenCV.

Al importarse, ultralytics ejecuta `_imshow = cv2.imshow` (ver
ultralytics/utils/patches.py). Las compilaciones `opencv-python-headless`
omiten las funciones de interfaz grafica, de modo que ese atributo no existe
y el import falla con:

    AttributeError: module 'cv2' has no attribute 'imshow'

Un servicio web no dispone de entorno grafico ni puede abrir ventanas, asi
que estas funciones nunca llegarian a usarse: se sustituyen por operaciones
nulas. La alternativa seria instalar la variante completa de OpenCV, que
arrastra libgl1 y unos 330 MB de controladores graficos inutiles en un
servidor sin pantalla.

Este modulo debe importarse ANTES que ultralytics.
"""

import logging as log

import cv2

# Funciones de interfaz grafica que ultralytics referencia o podria referenciar.
FUNCIONES_GRAFICAS = ("imshow", "waitKey", "destroyAllWindows", "namedWindow")


def _sin_interfaz_grafica(*args, **kwargs):
    """Sustituto inerte: en un servidor no hay ventanas que mostrar."""
    return None


def aplicar_compatibilidad() -> list:
    """Define los atributos ausentes y devuelve los que hubo que sustituir."""
    sustituidos = []
    for nombre in FUNCIONES_GRAFICAS:
        if not hasattr(cv2, nombre):
            setattr(cv2, nombre, _sin_interfaz_grafica)
            sustituidos.append(nombre)

    if sustituidos:
        log.info(
            "OpenCV headless detectado; funciones graficas sustituidas: %s",
            ", ".join(sustituidos),
        )
    return sustituidos


aplicar_compatibilidad()
