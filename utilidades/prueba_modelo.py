from ultralytics import YOLO
model = YOLO("xrays_evaluation_model_medium_v1.pt")
print(model.names)