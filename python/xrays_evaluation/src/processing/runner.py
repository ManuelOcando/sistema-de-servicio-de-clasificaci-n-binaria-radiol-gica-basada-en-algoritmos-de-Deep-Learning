import os
import time
from src.processing.image_utils.main import ImageUtils
from src.inference.yolo_classifier import YoloClassifier
from src.processing.explainability.gradcam import GradCam

# Ruta del modelo. Se toma de la variable de entorno XRAY_MODEL_PATH para que
# el servicio funcione sin cambios en local, en contenedor y en la nube, donde
# el directorio de trabajo no siempre es el mismo.
RUTA_MODELO_POR_DEFECTO = os.getenv(
    "XRAY_MODEL_PATH", "models/YOLO/xrays_evaluation_model_medium_v1.pt"
)


class XRayInferencePipeline:
    def __init__(self, model_path: str = None):
        model_path = model_path or RUTA_MODELO_POR_DEFECTO
        self.image_utils = ImageUtils()
        self.model_wrapper = YoloClassifier(model_path)
        self.explainer = GradCam()

        # mapping class
        self.class_map = {0: "Anomaly", 1: "Normal"}

    def run(self, base64_image: str) -> dict:
        """
        execute inference
        """
        t_start = time.time()

        # 1. decode image Imagen
        original_image = self.image_utils.decode_base64(base64_image)
        t_decoded = time.time()

        # 2. inference
        prediction = self.model_wrapper.predict(original_image)
        t_inference = time.time()

        # 3. HeatMap
        raw_heatmap = self.explainer.generate_heatmap(
            image=original_image,
            model=prediction["internal_model"],
            class_idx=prediction["class_id"],
        )

        # 4. overlay
        # 4.1 Create Colored Heatmap
        colored_heatmap = self.image_utils.apply_heatmap_colors(raw_heatmap, original_image.shape)
        heatmap_base64 = self.image_utils.encode_base64(colored_heatmap)

        # 4.2
        overlay_img = self.image_utils.overlay_heatmap(original_image, colored_heatmap)
        overlay_base64 = self.image_utils.encode_base64(overlay_img)
        t_end = time.time()

        # calculate latencies
        # pre-processing time
        preprocess_ms = (t_decoded - t_start) * 1000

        # inference time
        inference_ms = (t_inference - t_decoded) * 1000

        # GradCAM time
        explainability_ms = (t_end - t_inference) * 1000

        # total time
        total_latency_ms = (t_end - t_start) * 1000

        # 5. build response
        return {
            "prediction": {
                "label": self.class_map.get(prediction["class_id"], "Unknown"),
                "confidence": round(prediction["confidence"], 4),
                "class_id": prediction["class_id"]
            },
            "explainability": {
                "heatmap_base64": heatmap_base64,  # colors
                "overlay_base64": overlay_base64,  # overlay
                "description": "Red indicates high attention regions."
            },
            "performance": {
                "preprocess_time_ms": round(preprocess_ms, 2),
                "inference_time_ms": round(inference_ms, 2),
                "explainability_time_ms": round(explainability_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "model_used": "YOLO11m-cls"
            }
        }
