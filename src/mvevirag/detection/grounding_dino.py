"""Open-vocabulary object detection with Grounding DINO (Module 1).

Implements manuscript III.C (open-vocabulary detection, person crop extraction
for cross-view ReID, and EvidenceItem assembly for scoring) and III.F
(implementation stack: ``IDEA-Research/grounding-dino-tiny`` via
``GroundingDinoForObjectDetection`` + ``AutoProcessor`` under ``transformers``
4.57.x).

Implementation notes
--------------------

* ``torch.backends.cudnn.enabled = False`` is set before the model is moved
  to device, mirroring :mod:`mvevirag.embedding.siglip2`, because the target
  server's cuDNN build is incompatible with the convolution path exercised
  by Grounding DINO's backbone.
* The model runs in ``float32``. ``float16`` triggers an input type mismatch
  with the text-image fusion heads under ``transformers`` 4.57.x (smoke
  test), so the encoder stays on full precision.
* Detection runs on ``cuda:2`` by default: ``cuda:0`` is owned by the vLLM
  Docker container and ``cuda:1`` by SigLIP 2. CPU fallback is used when
  CUDA is unavailable.
* The forward pass executes under :func:`torch.no_grad` and the post-helping
  ``post_process_grounded_object_detection`` API uses the parameter name
  ``threshold`` (not ``box_threshold``) under ``transformers`` 4.57.x.

Public surface
--------------

* :class:`GroundingDINODetector` - load-once detector with single-image /
  multi-frame detection, person crop extraction and EvidenceItem assembly.
* :class:`Detection` - dataclass for one detection.
* :func:`build_prompt` - build a Grounding DINO text prompt from required
  entity labels.
* :data:`LABEL_MAP` - bidirectional ``human`` <-> ``person`` alias, mirroring
  the constants in :mod:`mvevirag.evaluation.axis_accuracy` /
  :mod:`mvevirag.evaluation.schema_ucr` so detection and evaluation share a
  single reconciliation map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

__all__ = [
    "GroundingDINODetector",
    "Detection",
    "build_prompt",
    "LABEL_MAP",
]

# Default model id per manuscript III.F.
_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Default device: vLLM owns cuda:0, SigLIP 2 owns cuda:1, GD owns cuda:2.
_DEFAULT_DEVICE = "cuda:2"
# Fallback text prompt when no required entity can be resolved.
_FALLBACK_PROMPT = "person."

#: Bidirectional alias between the AIHub annotation label (``"human"``) and
#: the detector / VLM label (``"person"``). Matching is case-insensitive
#: after lowercasing, so aliases are stored lowercase. Mirrors
#: :data:`mvevirag.evaluation.axis_accuracy.LABEL_MAP` and
#: :data:`mvevirag.evaluation.schema_ucr.LABEL_MAP` so the detector module is
#: the canonical home for label space reconciliation across the pipeline.
LABEL_MAP: dict[str, str] = {
    "human": "person",
    "person": "human",
}


@dataclass
class Detection:
    """Single detection returned by :class:`GroundingDINODetector`.

    Attributes:
        label: Lowercase detector text label (``"person"``, ``"helmet"``,
            ``"motorcycle"`` ...).
        confidence: Detector confidence in ``[0, 1]``.
        bbox: ``[x1, y1, x2, y2]`` in absolute pixel coordinates of the
            source frame (``1920 x 1080`` for the AIHub corpus).
        frame_id: 0-indexed sampled frame id this detection came from.
            Defaults to ``0`` for single-image :meth:`detect` calls;
            :meth:`detect_frames` / :meth:`process_view` overwrite it.
    """

    label: str
    confidence: float
    bbox: list[int]
    frame_id: int = 0


def build_prompt(required_entities: list[str]) -> str:
    """Build a Grounding DINO text prompt from required entity labels.

    Each label is wrapped in a trailing period and labels are space-joined,
    matching the smoke-test format ``"person. helmet. motorcycle."``. Empty
    or all-whitespace labels are dropped; the resulting label list is
    lowercased so the prompt matches Grounding DINO's training vocabulary.

    The function is leakage-safe: it consumes only the label list produced by
    :func:`mvevirag.scoring.query_type.required_entities_for_query`, never
    any annotation / gold field.

    Args:
        required_entities: Detector labels induced from the query text
            (e.g. ``["person", "helmet"]``).

    Returns:
        Period-separated prompt string ending in ``"."``. Falls back to
        ``"person."`` when no valid label remains (per manuscript III.C).
    """
    labels = [str(lbl).strip().lower() for lbl in (required_entities or [])]
    labels = [lbl for lbl in labels if lbl]
    if not labels:
        return _FALLBACK_PROMPT
    return " ".join(f"{lbl}." for lbl in labels)


class GroundingDINODetector:
    """Open-vocabulary object detector using Grounding DINO.

    Used for:

    * Object detection on sampled frame PNGs (returns
      :class:`Detection` instances).
    * Person crop extraction (PIL images) for SigLIP 2 ReID encoding
      (:mod:`mvevirag.embedding.siglip2`).
    * Evidence item assembly for :mod:`mvevirag.scoring.six_axes`.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        """Load the Grounding DINO model and processor.

        ``torch.backends.cudnn.enabled`` is disabled for server
        compatibility, the model is cast to ``float32`` and moved to
        ``device`` (falling back to CPU when CUDA is unavailable).

        Args:
            model_id: HuggingFace model id (default
                ``"IDEA-Research/grounding-dino-tiny"``).
            device: Target device (default ``"cuda:2"``; ``cuda:0`` is
                owned by the vLLM Docker, ``cuda:1`` by SigLIP 2).
        """
        torch.backends.cudnn.enabled = False  # server cuDNN compatibility
        self.model_id = model_id
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id
        ).to(device=self.device, dtype=self.dtype)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_image(image: str | Path | Image.Image) -> Image.Image:
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

    def _run_model(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float,
        text_threshold: float,
    ) -> list[Detection]:
        """Run a single forward pass and decode to a :class:`Detection` list.

        ``text_prompt`` is forwarded verbatim to the processor; callers are
        expected to build it via :func:`build_prompt` so it is never empty.
        """
        width, height = image.size  # PIL: (W, H)
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )
        # Cast only floating-point tensors to the model dtype; input_ids /
        # attention_mask / pixel_mask stay on their integral dtype.
        inputs = {
            k: v.to(
                device=self.device,
                dtype=self.dtype if v.is_floating_point() else v.dtype,
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],  # (H, W) per HF API
        )
        return self._decode_results(results, width=width, height=height)

    @staticmethod
    def _decode_results(
        results: list,
        width: int,
        height: int,
    ) -> list[Detection]:
        """Convert the GD post-processor output to a :class:`Detection` list.

        ``results[0]`` is expected to expose ``scores``, ``boxes``,
        ``text_labels`` (and ``labels``) as returned by
        ``post_process_grounded_object_detection``. Boxes are scaled to the
        original frame size by the post-processor and clipped defensively to
        ``[0, width]`` / ``[0, height]`` here.
        """
        if not results:
            return []
        result = results[0]
        if not isinstance(result, dict):
            return []
        scores = result.get("scores", [])
        boxes = result.get("boxes", [])
        text_labels = result.get("text_labels", [])

        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        boxes_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
        labels_list = (
            text_labels.tolist() if hasattr(text_labels, "tolist") else list(text_labels)
        )

        detections: list[Detection] = []
        n = min(len(scores_list), len(boxes_list))
        for idx in range(n):
            box = boxes_list[idx]
            try:
                coords = [int(round(float(c))) for c in list(box)[:4]]
            except (TypeError, ValueError):
                continue
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords
            # Defensive clipping so the bbox stays within the frame.
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            label_raw = labels_list[idx] if idx < len(labels_list) else ""
            label = str(label_raw).strip().lower() if label_raw else ""
            confidence = float(scores_list[idx])
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    frame_id=0,
                )
            )
        return detections

    # ------------------------------------------------------------------ #
    # Single-image detection
    # ------------------------------------------------------------------ #
    def detect(
        self,
        image: str | Image.Image,
        text_prompt: str,
        threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Detect objects in ``image`` given ``text_prompt``.

        Args:
            image: Path to an image file or :class:`PIL.Image.Image`.
            text_prompt: Period-separated labels, e.g.
                ``"person. helmet. motorcycle."``.
            threshold: Box confidence threshold (0.3 per manuscript).
            text_threshold: Text confidence threshold (0.25).

        Returns:
            List of :class:`Detection` instances. ``frame_id`` is left at
            the default ``0``; use :meth:`detect_frames` /
            :meth:`process_view` to populate it per sampled frame.

        Raises:
            FileNotFoundError: When ``image`` is a non-existent path.
            ValueError: When ``image`` cannot be opened or has zero area.
        """
        img = self._load_image(image)
        return self._run_model(
            img,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
        )

    # ------------------------------------------------------------------ #
    # Multi-frame detection
    # ------------------------------------------------------------------ #
    def detect_frames(
        self,
        frame_paths: list[str],
        text_prompt: str,
        frame_ids: list[int],
        threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Detect objects across multiple frames, tagging ``frame_id`` on each.

        Args:
            frame_paths: List of image file paths (one sampled view).
            text_prompt: Period-separated labels (see :meth:`detect`).
            frame_ids: Parallel list of 0-indexed sampled frame ids. When
                shorter than ``frame_paths`` the index position is used; when
                longer, the surplus is ignored.
            threshold: Box confidence threshold (0.3 per manuscript).
            text_threshold: Text confidence threshold (0.25).

        Returns:
            All :class:`Detection` instances across frames with their
            ``frame_id`` populated. Empty list when ``frame_paths`` is empty.

        Raises:
            FileNotFoundError: When any frame path does not exist.
            ValueError: When any frame cannot be opened or has zero area.
        """
        detections: list[Detection] = []
        if not frame_paths:
            return detections
        ids = list(frame_ids) if frame_ids is not None else []
        for idx, path in enumerate(frame_paths):
            frame_id = int(ids[idx]) if idx < len(ids) else idx
            frame_dets = self.detect(
                image=path,
                text_prompt=text_prompt,
                threshold=threshold,
                text_threshold=text_threshold,
            )
            for det in frame_dets:
                det.frame_id = frame_id
                detections.append(det)
        return detections

    # ------------------------------------------------------------------ #
    # Person crop extraction
    # ------------------------------------------------------------------ #
    def extract_person_crops(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> list[Image.Image]:
        """Extract PIL crops for ``person`` detections from ``image``.

        The bbox is converted to ints and clipped to the image bounds
        (``min = 0``, ``max = image.width`` / ``image.height``). Zero-area
        boxes are skipped. Crops are returned as RGB :class:`PIL.Image.Image`,
        ready for :meth:`mvevirag.embedding.siglip2.SigLIP2Encoder.encode_person_crops`.

        Args:
            image: Source :class:`PIL.Image.Image`.
            detections: Detections returned by :meth:`detect` /
                :meth:`detect_frames`. Only detections whose ``label`` is
                ``"person"`` are cropped.

        Returns:
            List of :class:`PIL.Image.Image` crops, in input order. Empty
            when ``image`` is degenerate, ``detections`` is empty, or no
            person detection has a positive-area bbox.
        """
        if image is None or 0 in image.size:
            return []
        width, height = image.size
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        crops: list[Image.Image] = []
        for det in detections or []:
            label = (det.label or "").strip().lower()
            if label != "person":
                continue
            if len(det.bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(c) for c in det.bbox[:4])
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(rgb.crop((x1, y1, x2, y2)))
        return crops

    # ------------------------------------------------------------------ #
    # EvidenceItem assembly
    # ------------------------------------------------------------------ #
    def build_evidence_item(
        self,
        sample_id: str,
        view_id: str,
        frame_paths: list[str],
        frame_ids: list[int],
        detections: list[Detection],
        frame_size: tuple[int, int],
        provenance: dict,
        time_span: tuple[float, float],
    ) -> dict:
        """Build an EvidenceItem dict for :mod:`mvevirag.scoring.six_axes`.

        Each :class:`Detection` becomes one ``entity_region`` entry with its
        ``frame_id`` wrapped in a singleton ``frame_ids`` list. The item
        ``frame_ids`` field is the union of sampled frame ids supplied by the
        caller. The output structure matches
        :func:`mvevirag.scoring.six_axes.observed_support`'s contract:

        ``{evidence_id, sample_id, view_id, frame_ids, entity_regions,
        frame_size, provenance, time_span}``.

        Args:
            sample_id: AIHub sample id (e.g. ``"fp_e1723"``).
            view_id: View identifier (``"c1"`` or ``"c2"``).
            frame_paths: Sampled frame paths; used only as a count reference
                and not opened here.
            frame_ids: 0-indexed sampled frame ids parallel to
                ``frame_paths``.
            detections: All detections for this view across all frames.
            frame_size: ``(width, height)`` of the source frame
                (``(1920, 1080)`` for the AIHub corpus).
            provenance: ``{cctv_camera, source_video, timestamp}`` dict.
                Missing keys default to empty strings.
            time_span: ``(start, end)`` seconds span of the sampled frames.

        Returns:
            EvidenceItem dict ready for
            :func:`mvevirag.scoring.six_axes.observed_support`.
        """
        _ = frame_paths  # accepted for API symmetry; not opened here
        safe_frame_ids = [int(fid) for fid in (frame_ids or [])]
        regions: list[dict] = []
        for det in detections or []:
            bbox = (
                [int(c) for c in det.bbox[:4]]
                if len(det.bbox) >= 4
                else []
            )
            regions.append(
                {
                    "label": str(det.label or ""),
                    "confidence": float(det.confidence),
                    "bbox": bbox,
                    "frame_ids": [int(det.frame_id)],
                }
            )
        if isinstance(frame_size, (list, tuple)) and len(frame_size) >= 2:
            fs = [int(frame_size[0]), int(frame_size[1])]
        else:
            fs = [0, 0]
        provenance_dict = dict(provenance or {})
        prov = {
            "cctv_camera": str(provenance_dict.get("cctv_camera", "")),
            "source_video": str(provenance_dict.get("source_video", "")),
            "timestamp": str(provenance_dict.get("timestamp", "")),
        }
        try:
            ts_start = float(time_span[0]) if time_span else 0.0
            ts_end = float(time_span[1]) if time_span and len(time_span) > 1 else 0.0
        except (TypeError, ValueError, IndexError):
            ts_start, ts_end = 0.0, 0.0
        evidence_id = f"{sample_id}_{view_id}"
        return {
            "evidence_id": evidence_id,
            "sample_id": str(sample_id),
            "view_id": str(view_id),
            "frame_ids": safe_frame_ids,
            "entity_regions": regions,
            "frame_size": fs,
            "provenance": prov,
            "time_span": [ts_start, ts_end],
        }

    # ------------------------------------------------------------------ #
    # End-to-end per-view orchestration
    # ------------------------------------------------------------------ #
    def process_view(
        self,
        frame_paths: list[str],
        frame_ids: list[int],
        sample_id: str,
        view_id: str,
        required_entities: list[str],
        frame_size: tuple[int, int] = (1920, 1080),
        provenance: dict | None = None,
        time_span: tuple[float, float] | None = None,
    ) -> tuple[dict, list[Image.Image]]:
        """Process a single view end-to-end.

        Convenience wrapper that: (1) builds the text prompt from
        ``required_entities`` via :func:`build_prompt`; (2) runs detection
        per frame, tagging each detection with its ``frame_id``; (3) extracts
        person crops from every frame in which a person was detected; and
        (4) assembles an EvidenceItem ready for
        :func:`mvevirag.scoring.six_axes.observed_support`.

        Args:
            frame_paths: Sampled frame paths (one view).
            frame_ids: Parallel list of 0-indexed sampled frame ids.
            sample_id: AIHub sample id (e.g. ``"fp_e1723"``).
            view_id: View identifier (``"c1"`` or ``"c2"``).
            required_entities: Detector labels induced from the query text
                (leakage-safe; see
                :func:`mvevirag.scoring.query_type.required_entities_for_query`).
            frame_size: ``(width, height)`` of the source frame
                (default ``(1920, 1080)`` for the AIHub corpus).
            provenance: ``{cctv_camera, source_video, timestamp}`` dict.
                When ``None`` empty strings are used.
            time_span: ``(start, end)`` seconds span. When ``None``
                ``(0.0, 0.0)`` is used.

        Returns:
            ``(evidence_item, person_crops)``. The ``evidence_item`` is
            ready to pass to
            :func:`mvevirag.scoring.six_axes.observed_support`; the
            ``person_crops`` list can be passed to
            :meth:`mvevirag.embedding.siglip2.SigLIP2Encoder.encode_person_crops`.
        """
        text_prompt = build_prompt(required_entities)
        all_detections: list[Detection] = []
        person_crops: list[Image.Image] = []
        ids = list(frame_ids) if frame_ids is not None else []
        for idx, path in enumerate(frame_paths or []):
            frame_id = int(ids[idx]) if idx < len(ids) else idx
            try:
                image = self._load_image(path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"process_view failed to load frame {path}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"process_view failed to load frame {path}: {exc}") from exc
            frame_dets = self._run_model(
                image,
                text_prompt=text_prompt,
                threshold=0.3,
                text_threshold=0.25,
            )
            for det in frame_dets:
                det.frame_id = frame_id
                all_detections.append(det)
            frame_persons = [
                d for d in frame_dets if (d.label or "").strip().lower() == "person"
            ]
            person_crops.extend(self.extract_person_crops(image, frame_persons))
        evidence_item = self.build_evidence_item(
            sample_id=sample_id,
            view_id=view_id,
            frame_paths=frame_paths,
            frame_ids=frame_ids,
            detections=all_detections,
            frame_size=frame_size,
            provenance=provenance or {},
            time_span=time_span or (0.0, 0.0),
        )
        return evidence_item, person_crops
