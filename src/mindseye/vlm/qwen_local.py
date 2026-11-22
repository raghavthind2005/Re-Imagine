"""Local Qwen3-VL backend (``Qwen/Qwen3-VL-8B-Instruct`` and ``-4B-Instruct``).

Designed for Kaggle T4s: fp16 weights, spread over the available GPUs with
``device_map="auto"`` (the 8B model is 17.5 GB and needs both T4s in fp16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mindseye.utils.cache import DiskCache, cache_key
from mindseye.vlm.base import Image, VLMBackend, VLMResponse, image_fingerprint


def fingerprint(vlm_cfg: dict[str, Any]) -> dict[str, Any]:
    """Cache fingerprint of a ``QwenLocal`` built from this config, without loading it."""
    return {
        "dtype": vlm_cfg.get("dtype", "float16"),
        "load_in_4bit": vlm_cfg.get("load_in_4bit", False),
        "max_pixels": vlm_cfg.get("max_pixels"),
        "resize": "pil-bicubic-v1",
    }


class QwenLocal(VLMBackend):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        dtype: str = "float16",
        load_in_4bit: bool = False,
        max_pixels: int | None = None,
        max_memory: dict[int, str] | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(cache)
        import torch
        import transformers
        from transformers import AutoProcessor

        # The model card names Qwen3VLForConditionalGeneration; older or newer
        # transformers releases may only expose it through the Auto class.
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForImageTextToText

        self.model_id = model_id
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": getattr(torch, dtype)}
        if max_memory is not None:
            # Leaves headroom on the GPU that holds the vision tower, whose
            # attention over many frames needs several GB of working memory.
            kwargs["max_memory"] = max_memory
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        self.model = model_cls.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._torch = torch

    def config_fingerprint(self) -> dict[str, Any]:
        return fingerprint(
            {"dtype": self.dtype, "load_in_4bit": self.load_in_4bit, "max_pixels": self.max_pixels}
        )

    def _to_pil(self, image: Image):
        """Load an image and shrink it to at most ``max_pixels``, keeping its aspect ratio.

        Resizing is done here, not left to the processor: transformers 5.19 ignored
        a per-image ``max_pixels`` and passed frames through at native size.
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil = PILImage.fromarray(image)
        else:
            pil = PILImage.open(Path(image)).convert("RGB")
        if self.max_pixels is not None and pil.width * pil.height > self.max_pixels:
            scale = (self.max_pixels / (pil.width * pil.height)) ** 0.5
            size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
            pil = pil.resize(size, PILImage.Resampling.BICUBIC)
        return pil

    def _inputs(self, images: list[Image], prompt: str):
        content: list[dict[str, Any]] = [
            {"type": "image", "image": self._to_pil(im)} for im in images
        ]
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    def yes_no(self, images: list[Image], prompt: str, *, prompt_version: str) -> dict[str, float]:
        """Probability of "yes" and "no" as the first answer token, from one forward pass.

        ``p_no`` is renormalised over the two answers; ``mass`` is the share of the
        first-token probability they take together. Cached like ``generate``.
        """
        key = cache_key(
            kind="yes_no-v1",
            model_id=self.model_id,
            backend=self.config_fingerprint(),
            prompt_version=prompt_version,
            prompt=prompt,
            images=[image_fingerprint(im) for im in images],
        )
        if self.cache is not None and (hit := self.cache.get_json(key)) is not None:
            return hit
        torch = self._torch
        tok = self.processor.tokenizer
        ids = {
            answer: sorted({tok.encode(v, add_special_tokens=False)[0] for v in variants})
            for answer, variants in (("yes", ("yes", "Yes", "YES")), ("no", ("no", "No", "NO")))
        }
        with torch.inference_mode():
            logits = self.model(**self._inputs(images, prompt)).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes, p_no = (float(probs[ids[a]].sum()) for a in ("yes", "no"))
        out = {"p_no": p_no / (p_yes + p_no), "mass": p_yes + p_no}
        if self.cache is not None:
            self.cache.put_json(key, out)
        return out

    def option_probs(
        self, images: list[Image], prompt: str, letters: str, *, prompt_version: str
    ) -> dict[str, float]:
        """Probability of each option letter as the first answer token, renormalised.

        Used as the answer confidence for multiple-choice questions. Cached.
        """
        key = cache_key(
            kind="option_probs-v1",
            model_id=self.model_id,
            backend=self.config_fingerprint(),
            prompt_version=prompt_version,
            prompt=prompt,
            letters=letters,
            images=[image_fingerprint(im) for im in images],
        )
        if self.cache is not None and (hit := self.cache.get_json(key)) is not None:
            return hit
        torch = self._torch
        tok = self.processor.tokenizer
        ids = {x: tok.encode(x, add_special_tokens=False)[0] for x in letters}
        with torch.inference_mode():
            logits = self.model(**self._inputs(images, prompt)).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        raw = {x: float(probs[i]) for x, i in ids.items()}
        total = sum(raw.values())
        out = {x: v / total for x, v in raw.items()} | {"_mass": total}
        if self.cache is not None:
            self.cache.put_json(key, out)
        return out

    def _generate(
        self, images: list[Image], prompt: str, *, max_tokens: int, temperature: float, seed: int
    ) -> VLMResponse:
        torch = self._torch
        inputs = self._inputs(images, prompt)
        torch.manual_seed(seed)
        sample = temperature > 0
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": sample}
        if sample:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        n_prompt = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0]
        return VLMResponse(
            text=text.strip(),
            raw={"prompt_tokens": int(n_prompt), "new_tokens": int(out.shape[1] - n_prompt)},
        )
