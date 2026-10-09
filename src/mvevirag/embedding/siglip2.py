"""SigLIP 2 multilingual text/image/ReID encoder (Module 2).

Implements manuscript III.C (query-frame relevance, cross-view person identity
matching) and III.F (implementation stack: ``google/siglip2-base-patch16-384``
as the vision-language embedding encoder). Query strings and frame PNGs are
projected into a shared 768-dim space so that, after L2 normalisation, cosine
similarity drives both the FAISS ``IndexFlatIP`` retriever (Module 5) and the
ReID proxy used for cross-view person matching.

Implementation notes
--------------------

* The model is loaded via :func:`transformers.AutoModel.from_pretrained`
  rather than importing ``Siglip2Model`` directly. Under ``transformers``
  4.57.6 the direct import raises a size-mismatch error when the patched
  checkpoint resolves to the legacy ``SiglipModel`` class; ``AutoModel``
  routes to ``SiglipModel`` correctly (the smoke-test-confirmed model type),
  and exposes ``get_text_features`` / ``get_image_features`` directly.
* ``torch.backends.cudnn.enabled = False`` is set before the model is moved
  to device, because the target server's cuDNN build is incompatible with
  the convolution path exercised by SigLIP 2's patch embedding.
* The encoder runs on ``cuda:1`` by default because the vLLM Docker container
  owns ``cuda:0``; the device is overridable via the ``device`` argument and
  falls back to CPU when CUDA is unavailable.
* Model inference and returned embeddings are fixed to float32. Embeddings are
  L2-normalised NumPy arrays so that
  :func:`numpy.inner` (and FAISS ``IndexFlatIP``) computes cosine similarity
  directly without extra normalisation at the call site.

Public surface
--------------

* :class:`SigLIP2Encoder` - load-once encoder with text/image/ReID/batch
  helpers plus a cosine-similarity convenience method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

__all__ = ["SigLIP2Encoder"]

# SigLIP 2 base multilingual embedding dimension (manuscript III.F).
_DEFAULT_DIM = 768
# Default model identifier per manuscript III.F.
_DEFAULT_MODEL_ID = "google/siglip2-base-patch16-384"
# Default device: vLLM Docker occupies cuda:0, encoder runs on cuda:1.
_DEFAULT_DEVICE = "cuda:1"


class SigLIP2Encoder:
    """SigLIP 2 multilingual encoder for text/image/ReID embeddings.

    Used for:

    * Query-frame relevance (cosine similarity) - :meth:`encode_text` vs
      :meth:`encode_image`.
    * Cross-view person identity matching (ReID proxy) -
      :meth:`encode_person_crops`.
    * FAISS ``IndexFlatIP`` construction (:mod:`mvevirag.index.faiss_index`).

    All embeddings are L2-normalised float32 NumPy arrays of length 768 so
    inner product == cosine similarity.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        """Load the SigLIP 2 model and processor.

        ``torch.backends.cudnn.enabled`` is disabled for server
        compatibility, the model is kept in ``float32`` and moved to
        ``device`` (falling back to CPU when CUDA is unavailable).

        Args:
            model_id: HuggingFace model id (default
                ``"google/siglip2-base-patch16-384"``).
            device: Target device (default ``"cuda:1"``; ``"cuda:0"`` is
                occupied by the vLLM Docker).
        """
        torch.backends.cudnn.enabled = False  # server cuDNN compatibility
        self.model_id = model_id
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(
            device=self.device, dtype=self.dtype
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalise along the last dim, guarding against zero vectors."""
        norm = tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return tensor / norm

    @staticmethod
    def _pooled_features(
        features: torch.Tensor | BaseModelOutputWithPooling,
    ) -> torch.Tensor:
        """Return the pooled feature tensor across Transformers 4.x and 5.x."""
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, BaseModelOutputWithPooling):
            if not isinstance(features.pooler_output, torch.Tensor):
                raise TypeError("feature output pooler_output must be a torch.Tensor")
            return features.pooler_output
        raise TypeError(
            "feature output must be a torch.Tensor or BaseModelOutputWithPooling, "
            f"got {type(features).__name__}"
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Move an embedding tensor to CPU float32 NumPy."""
        return tensor.detach().cpu().float().numpy()

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        """Load a PIL image from a path or accept a PIL image directly.

        Args:
            image: File path (``str`` / ``Path``) or a
                :class:`PIL.Image.Image`.

        Returns:
            An RGB :class:`PIL.Image.Image`.

        Raises:
            FileNotFoundError: When ``image`` is a path that does not exist.
            ValueError: When the image cannot be opened or has zero area.
        """
        if isinstance(image, Image.Image):
            img = image
        else:
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            try:
                img = Image.open(path)
            except OSError as exc:
                raise ValueError(f"Failed to open image {path}: {exc}") from exc
        img = img.convert("RGB")
        if 0 in img.size:
            raise ValueError(f"Image has zero area: {img.size}")
        return img

    # ------------------------------------------------------------------ #
    # Single-item encoding
    # ------------------------------------------------------------------ #
    def encode_text(self, query: str) -> np.ndarray:
        """Encode a query string to a 768-dim L2-normalised embedding.

        Args:
            query: Query text (multilingual).

        Returns:
            1-D float32 array of length 768.

        Raises:
            ValueError: When ``query`` is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.encode_batch_texts([query])[0]

    def encode_image(self, image: str | Path | Image.Image) -> np.ndarray:
        """Encode a frame image to a 768-dim L2-normalised embedding.

        Args:
            image: Path to a PNG/JPG frame or a :class:`PIL.Image.Image`.

        Returns:
            1-D float32 array of length 768.

        Raises:
            FileNotFoundError: When ``image`` is a path that does not exist.
            ValueError: When the image cannot be opened or has zero area.
        """
        img = self._load_image(image)
        return self.encode_batch_images([img], batch_size=1)[0]

    def encode_person_crops(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """Encode person crop images to a list of 768-dim embeddings (ReID).

        Reuses the image-encoding path so person crops and full-frame
        embeddings live in the same space, enabling cross-view person
        matching via :meth:`cosine_similarity` (manuscript III.C, ReID
        proxy).

        Args:
            crops: List of :class:`PIL.Image.Image` person crops.

        Returns:
            List of 1-D float32 arrays, one per crop, in input order.

        Raises:
            ValueError: When ``crops`` is empty.
            TypeError: When any element is not a :class:`PIL.Image.Image`.
        """
        if not crops:
            raise ValueError("crops must be a non-empty list")
        valid_crops: list[Image.Image] = []
        for crop in crops:
            if not isinstance(crop, Image.Image):
                raise TypeError("each crop must be a PIL.Image.Image")
            valid_crops.append(crop.convert("RGB"))
        embeddings = self.encode_batch_images(valid_crops, batch_size=32)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    # ------------------------------------------------------------------ #
    # Batch encoding
    # ------------------------------------------------------------------ #
    def encode_batch_texts(self, queries: list[str]) -> np.ndarray:
        """Encode multiple query strings to an ``(N, 768)`` array.

        Args:
            queries: List of query strings.

        Returns:
            Float32 array of shape ``(N, 768)``, L2-normalised per row.

        Raises:
            ValueError: When ``queries`` is empty or any query is empty.
        """
        if not queries:
            raise ValueError("queries must be a non-empty list")
        for q in queries:
            if not q or not q.strip():
                raise ValueError("each query must be a non-empty string")
        inputs = self.processor(
            text=list(queries),
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        # Cast only floating-point tensors to float32; input_ids / attention_mask
        # stay on their integral dtype.
        inputs = {
            k: v.to(
                device=self.device,
                dtype=self.dtype if v.is_floating_point() else v.dtype,
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        features = self._pooled_features(features)
        features = self._l2_normalize(features.float())
        return self._to_numpy(features)

    def encode_batch_images(
        self,
        image_paths: list[str | Path | Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode multiple frame images to an ``(N, 768)`` array.

        Args:
            image_paths: List of file paths or :class:`PIL.Image.Image`.
            batch_size: Number of images per forward pass (default 32).

        Returns:
            Float32 array of shape ``(N, 768)``, L2-normalised per row.

        Raises:
            FileNotFoundError: When any path is missing.
            ValueError: When ``image_paths`` is empty or any image is bad.
        """
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")
        images = [self._load_image(item) for item in image_paths]
        if batch_size <= 0:
            batch_size = len(images)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    device=self.device, dtype=self.dtype
                )
                features = self.model.get_image_features(pixel_values=pixel_values)
                features = self._pooled_features(features)
                features = self._l2_normalize(features.float())
                outputs.append(self._to_numpy(features))
        if not outputs:
            return np.zeros((0, _DEFAULT_DIM), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    # ------------------------------------------------------------------ #
    # Similarity
    # ------------------------------------------------------------------ #
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings.

        Both vectors are re-normalised defensively so the result lives in
        ``[-1, 1]`` even when the inputs were not pre-normalised.

        Args:
            a: 1-D embedding array.
            b: 1-D embedding array of the same length.

        Returns:
            Cosine similarity in ``[-1, 1]``. ``0.0`` is returned when either
            vector is all zeros (cosine undefined).
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
