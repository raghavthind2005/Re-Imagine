"""SAM 3 wrapper: open-vocabulary instance masks for a list of text labels.

Weights: facebook/sam3 (gated, SAM License). Uses the Transformers classes named
on the model card (``Sam3Model``, ``Sam3Processor``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Object categories that VSI-Bench questions ask about most often (taken from the
# question text of the counting, size, distance and appearance-order tasks).
VSI_OBJECTS = [
    "door", "table", "chair", "window", "tv", "sofa", "trash bin", "bed", "backpack",
    "refrigerator", "lamp", "pillow", "keyboard", "printer", "monitor", "telephone", "mirror",
    "bookshelf", "ceiling light", "toilet", "computer mouse", "basket", "nightstand",
    "microwave", "clock", "stool", "kettle", "laptop", "plant", "stove", "towel", "heater",
    "closet", "whiteboard", "blanket", "suitcase", "radiator", "washer", "bathtub", "fireplace",
]  # fmt: skip


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    mask: np.ndarray  # (H, W) bool
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

    @property
    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        return float(xs.mean()), float(ys.mean())

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class SAM3Runner:
    def __init__(
        self, weights: str = "facebook/sam3", device: str = "cuda", dtype: str = "float16"
    ):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = Sam3Model.from_pretrained(weights, torch_dtype=self.dtype).eval().to(device)
        self.processor = Sam3Processor.from_pretrained(weights)
        self._text_cache: dict[str, dict] = {}

    def _text(self, label: str) -> dict:
        """Tokenised prompt and its text embedding, computed once per label."""
        if label not in self._text_cache:
            tok = self.processor(text=label, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                emb = self.model.get_text_features(
                    input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
                )
            self._text_cache[label] = {
                "attention_mask": tok["attention_mask"],
                "text_embeds": emb,
            }
        return self._text_cache[label]

    def detect(
        self,
        image: np.ndarray,
        labels: list[str],
        *,
        threshold: float = 0.5,
        min_area_fraction: float = 0.002,
    ) -> list[Detection]:
        """All instances of each label in one image, above the score and size thresholds.

        The image goes through the vision encoder once; each label then costs only a
        decoder pass.
        """
        from PIL import Image

        torch = self._torch
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        sizes = inputs["original_sizes"].tolist()
        min_area = min_area_fraction * image.shape[0] * image.shape[1]
        found: list[Detection] = []
        with torch.no_grad():
            vision = self.model.get_vision_features(
                pixel_values=inputs["pixel_values"].to(self.dtype)
            )
            for label in labels:
                outputs = self.model(vision_embeds=vision, **self._text(label))
                res = self.processor.post_process_instance_segmentation(
                    outputs, threshold=threshold, mask_threshold=0.5, target_sizes=sizes
                )[0]
                for mask, box, score in zip(res["masks"], res["boxes"], res["scores"], strict=True):
                    m = mask.detach().cpu().numpy().astype(bool)
                    if m.sum() >= min_area:
                        found.append(
                            Detection(label, float(score), m, tuple(float(v) for v in box.tolist()))
                        )
        return found
