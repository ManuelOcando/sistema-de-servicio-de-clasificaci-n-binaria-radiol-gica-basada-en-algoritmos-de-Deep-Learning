from ultralytics import YOLO
model = YOLO("nombre_del_archivo.pt")
print(model.names)