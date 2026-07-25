from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    label: str = Field(..., description="Etiqueta predicha (Normal o Anomaly).")
    confidence: float = Field(..., description="Nivel de confianza de la predicción (0-1).")
    class_id: int = Field(..., description="ID numérico de la clase.")


class ExplainabilityData(BaseModel):
    heatmap_base64: str = Field(..., description="Imagen del mapa de calor (solo colores) en Base64.")
    overlay_base64: str = Field(..., description="Imagen original superpuesta con el mapa de calor en Base64.")
    description: str = Field(..., description="Descripción textual de la explicabilidad.")


class PerformanceData(BaseModel):
    preprocess_time_ms: float = Field(..., description="Tiempo tomado en decodificar y preparar la imagen.")
    inference_time_ms: float = Field(..., description="Tiempo tomado por el modelo YOLO en inferir.")
    explainability_time_ms: float = Field(..., description="Tiempo tomado en generar GradCAM.")
    total_latency_ms: float = Field(..., description="Latencia total del servicio.")
    model_used: str = Field(..., description="Nombre del modelo utilizado.")


class XRayInput(BaseModel):
    """
    structure input.
    """
    image_base64: str = Field(..., description="Cadena Base64 de la imagen de Rayos X a analizar.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
            }
        }
    }


class XRayOutput(BaseModel):
    """
    structure output API.
    """
    prediction: PredictionData
    explainability: ExplainabilityData
    performance: PerformanceData

