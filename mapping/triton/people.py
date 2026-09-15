"""Person masks from torchvision's COCO-trained Mask R-CNN."""

import numpy as np


class PersonSegmenter:
    """Person masks at the input image's resolution."""

    def __init__(self, score_threshold=0.5):
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        self._torch = torch
        self._score_threshold = score_threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        ).to(self._device).eval()

    def mask(self, image_bgr):
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        image = self._torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).to(
            self._device
        )
        with self._torch.inference_mode():
            prediction = self._model([image])[0]
        people = (prediction["labels"] == 1) & (
            prediction["scores"] >= self._score_threshold
        )
        if not bool(people.any()):
            return np.zeros(image_bgr.shape[:2], dtype=bool)
        masks = prediction["masks"][people, 0] >= 0.5
        return masks.any(dim=0).cpu().numpy()
