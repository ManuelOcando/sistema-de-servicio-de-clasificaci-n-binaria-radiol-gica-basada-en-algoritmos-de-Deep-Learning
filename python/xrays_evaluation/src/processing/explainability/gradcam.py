import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.processing.interfaces.main import IExplainer


class GradCam(IExplainer):
    """
    Grad-CAM (Selvaraju et al., 2017).

    Pondera los mapas de activación de la última capa convolucional por el
    gradiente del logit de la clase predicha, de modo que el mapa resultante
    es específico de la clase: explicar "Anomaly" produce un mapa distinto
    al de "Normal".

    El forward se ejecuta con el mismo preprocesamiento que aplica Ultralytics
    en inferencia (RGB, Resize al lado corto, CenterCrop, escala 1/255), para
    que la explicación corresponda exactamente a la entrada que vio el modelo
    al clasificar.
    """

    def __init__(self, imgsz: int = 224):
        self.imgsz = imgsz
        self.activations = None
        self.gradients = None

    # --- captura de activaciones y gradientes ---
    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        self.gradients = grad

    def _find_last_conv_layer(self, model):
        """
        Última capa convolucional del grafo. Se prefiere el bloque completo
        (Conv2d + BatchNorm + activación) sobre el Conv2d desnudo, porque
        Grad-CAM opera sobre activaciones post-activación.
        """
        last_block, last_conv = None, None
        for module in model.modules():
            children = list(module.children())
            if any(isinstance(c, nn.Conv2d) for c in children) and len(children) > 1:
                last_block = module
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_block if last_block is not None else last_conv

    # --- preprocesamiento equivalente a ultralytics.classify_transforms ---
    def _preprocess(self, image: np.ndarray) -> tuple:
        """
        Replica Resize(lado corto -> imgsz) + CenterCrop(imgsz) + ToTensor().
        Devuelve el tensor y la geometría necesaria para reproyectar el mapa
        sobre la imagen original.
        """
        h, w = image.shape[:2]
        scale = self.imgsz / min(h, w)
        nh, nw = round(h * scale), round(w * scale)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        top = (nh - self.imgsz) // 2
        left = (nw - self.imgsz) // 2
        cropped = resized[top:top + self.imgsz, left:left + self.imgsz]

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor, (nh, nw, top, left)

    def _project_to_original(self, cam: np.ndarray, geometry: tuple, shape: tuple) -> np.ndarray:
        """
        Reproyecta el mapa (imgsz x imgsz) sobre el lienzo original. Las zonas
        recortadas por el CenterCrop quedan en cero: el modelo nunca las vio,
        así que atribuirles atención sería incorrecto.
        """
        nh, nw, top, left = geometry
        h, w = shape[:2]

        cam_crop = cv2.resize(cam, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((nh, nw), dtype=np.float32)
        canvas[top:top + self.imgsz, left:left + self.imgsz] = cam_crop

        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)

    def generate_heatmap(self, image: np.ndarray, model: any, target_layer=None,
                         class_idx: int = None) -> np.ndarray:
        """
        Genera el mapa de calor Grad-CAM para la clase indicada.

        :param image: imagen original en BGR (tal como la entrega OpenCV).
        :param model: nn.Module subyacente del modelo YOLO.
        :param target_layer: capa objetivo; por defecto la última convolucional.
        :param class_idx: clase a explicar; por defecto la predicha por el modelo.
        :return: mapa normalizado [0, 1] con las dimensiones de la imagen original.
        """
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
            if target_layer is None:
                raise ValueError("No se encontró una capa Conv2d en el modelo para generar el Heatmap.")

        device = next(model.parameters()).device
        input_tensor, geometry = self._preprocess(image)
        input_tensor = input_tensor.to(device)

        # Todos los parámetros del modelo llegan congelados (requires_grad=False).
        # Habilitar el gradiente en la entrada basta para construir el grafo de
        # autograd y así poder retropropagar hasta las activaciones objetivo.
        input_tensor.requires_grad_(True)

        self.activations, self.gradients = None, None
        handle = target_layer.register_forward_hook(self._forward_hook)

        try:
            with torch.enable_grad():
                logits = model(input_tensor)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                logits = logits.reshape(1, -1)

                if class_idx is None:
                    class_idx = int(logits.argmax(dim=1).item())

                model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
        finally:
            handle.remove()

        if self.activations is None:
            raise RuntimeError("El hook no capturó activaciones.")
        if self.gradients is None:
            raise RuntimeError("El hook no capturó gradientes; no se pudo construir el grafo de autograd.")

        # Pesos de Grad-CAM: gradiente promediado sobre la dimensión espacial.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combinación lineal de los mapas de activación + ReLU: solo interesa
        # la evidencia que empuja hacia la clase, no la que la contradice.
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)

        # Normalización Min-Max
        min_val, max_val = torch.min(cam), torch.max(cam)
        if max_val - min_val > 0:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = torch.zeros_like(cam)

        cam = cam.detach().cpu().numpy().astype(np.float32)

        return self._project_to_original(cam, geometry, image.shape)
