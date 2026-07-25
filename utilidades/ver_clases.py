import os
from ultralytics import YOLO

def buscar_archivo(nombre):
    for raiz, dirs, archivos in os.walk(os.getcwd()):
        if nombre in archivos:
            return os.path.join(raiz, nombre)
    return None

# El nombre del archivo que vemos en tu VS Code
nombre_archivo = "xrays_evaluation_model_medium_v1.pt"
ruta_encontrada = buscar_archivo(nombre_archivo)

if ruta_encontrada:
    print(f"Archivo encontrado en: {ruta_encontrada}")
    try:
        model = YOLO(ruta_encontrada)
        print("\n--- ¡ÉXITO! CLASES MÉDICAS ---")
        print(model.names)
        print("-----------------------------\n")
    except Exception as e:
        print(f"Error al cargar: {e}")
else:
    print(f"No se encontró el archivo '{nombre_archivo}' en ninguna carpeta.")