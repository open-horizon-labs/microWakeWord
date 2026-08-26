"""Pure scoring primitives for the offline Kizz phoneme/CTC teacher.

The optional Wav2Vec2 implementation lives behind :func:`load_hf_teacher` and
is intentionally not imported at module import time.  This keeps the scoring
contract small, deterministic, and testable without a model download.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
CANONICAL_PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")
COLLISION_PHONES = {
    "hifi_kiss": ("h", "aɪ", "f", "aɪ", "k", "ɪ", "s"),
    "hifi_kids": ("h", "aɪ", "f", "aɪ", "k", "ɪ", "d", "z"),
    "highfive_kizz": ("h", "aɪ", "f", "aɪ", "v", "k", "ɪ", "z"),
    "highfive_kiss": ("h", "aɪ", "f", "aɪ", "v", "k", "ɪ", "s"),
    "highfive_kids": ("h", "aɪ", "f", "aɪ", "v", "k", "ɪ", "d", "z"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _logaddexp(a: float, b: float) -> float:
    if not math.isfinite(a):
        return b
    if not math.isfinite(b):
        return a
    return float(np.logaddexp(a, b))


def ctc_log_probability(
    log_probs: np.ndarray, tokens: Sequence[int], *, blank_id: int
) -> float:
    """Return the CTC forward log probability of one token sequence.

    This is the constrained CTC sum, not greedy decoding.  It accepts frame
    log probabilities shaped ``[frames, vocabulary]`` and uses a tiny Python
    DP so the exact contract can be tested independently of PyTorch.
    """
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("log_probs must be a non-empty [frames, vocabulary] array")
    labels = tuple(int(token) for token in tokens)
    if not labels:
        return float(np.sum(values[:, blank_id]))
    extended = (int(blank_id),) + sum(((token, int(blank_id)) for token in labels), ())
    states = np.full(len(extended), -np.inf, dtype=np.float64)
    states[0] = values[0, blank_id]
    states[1] = values[0, labels[0]]
    for frame in values[1:]:
        next_states = np.full_like(states, -np.inf)
        for state, token in enumerate(extended):
            total = states[state]
            if state:
                total = _logaddexp(total, states[state - 1])
            if state > 1 and token != blank_id and token != extended[state - 2]:
                total = _logaddexp(total, states[state - 2])
            next_states[state] = total + frame[token]
        states = next_states
    return _logaddexp(float(states[-1]), float(states[-2]))


def ctc_log_probability_batch(
    log_probs: np.ndarray, tokens: Sequence[int], *, blank_id: int
) -> np.ndarray:
    """Vectorized CTC forward probabilities for equal-length windows.

    Continuous qualification evaluates thousands of overlapping windows per
    audio chunk.  Calling the scalar reference DP for every window makes the
    fixed 100-hour gate take days.  This implementation preserves the same
    recurrence while vectorizing only across independent windows; the scalar
    implementation remains the small reference contract used by tests.
    """
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError(
            "log_probs must be a non-empty [windows, frames, vocabulary] array"
        )
    labels = tuple(int(token) for token in tokens)
    if not labels:
        return np.sum(values[:, :, blank_id], axis=1)
    extended = (int(blank_id),) + sum(
        ((token, int(blank_id)) for token in labels), ()
    )
    if min(extended) < 0 or max(extended) >= values.shape[2]:
        raise ValueError("CTC token ID is outside the supplied vocabulary")
    token_indexes = np.asarray(extended, dtype=np.int64)
    states = np.full((len(values), len(extended)), -np.inf, dtype=np.float64)
    states[:, 0] = values[:, 0, blank_id]
    states[:, 1] = values[:, 0, labels[0]]
    skip_allowed = np.asarray(
        [
            state > 1
            and token != blank_id
            and token != extended[state - 2]
            for state, token in enumerate(extended)
        ],
        dtype=bool,
    )
    for frame_index in range(1, values.shape[1]):
        next_states = states.copy()
        next_states[:, 1:] = np.logaddexp(
            next_states[:, 1:], states[:, :-1]
        )
        next_states[:, 2:] = np.where(
            skip_allowed[None, 2:],
            np.logaddexp(next_states[:, 2:], states[:, :-2]),
            next_states[:, 2:],
        )
        states = next_states + values[:, frame_index, token_indexes]
    return np.logaddexp(states[:, -1], states[:, -2])


def ctc_fit(log_probs: np.ndarray, tokens: Sequence[int], *, blank_id: int) -> float:
    """Length-normalized CTC fit; higher is better."""
    return ctc_log_probability(log_probs, tokens, blank_id=blank_id) / max(
        1, len(tokens)
    )


@dataclass(frozen=True)
class WindowScore:
    start_frame: int
    end_frame: int
    canonical_fit: float
    collision_fit: float
    collision_margin: float

    @property
    def eligible(self) -> bool:
        return math.isfinite(self.canonical_fit)


def score_window(
    log_probs: np.ndarray,
    *,
    canonical_tokens: Sequence[int],
    collision_tokens: Iterable[Sequence[int]],
    blank_id: int,
    start_frame: int = 0,
) -> WindowScore:
    canonical = ctc_fit(log_probs, canonical_tokens, blank_id=blank_id)
    collisions = [
        ctc_fit(log_probs, tokens, blank_id=blank_id) for tokens in collision_tokens
    ]
    collision = max(collisions, default=-math.inf)
    return WindowScore(
        start_frame=start_frame,
        end_frame=start_frame + len(log_probs),
        canonical_fit=canonical,
        collision_fit=collision,
        collision_margin=canonical - collision,
    )


def best_window_score(
    log_probs: np.ndarray,
    *,
    canonical_tokens: Sequence[int],
    collision_tokens: Iterable[Sequence[int]],
    blank_id: int,
    window_lengths: Sequence[int],
    hop: int,
    beta: float,
) -> WindowScore:
    """Score sliding windows, requiring canonical fit and margin ``>= beta``.

    ``canonical_fit`` remains the thresholded score.  The collision margin is
    a separate deterministic guard; combining them into one learned or tuned
    score would make the operating point ambiguous.
    """
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or hop <= 0 or not window_lengths:
        raise ValueError("invalid log-probability window configuration")
    candidates: list[WindowScore] = []
    seen_lengths = set()
    for requested_length in window_lengths:
        length = min(int(requested_length), len(values))
        if length <= 0:
            raise ValueError("window lengths must be positive")
        if length in seen_lengths:
            continue
        seen_lengths.add(length)
        starts = list(range(0, len(values) - length + 1, hop))
        tail = len(values) - length
        if not starts or starts[-1] != tail:
            starts.append(tail)
        for start in starts:
            candidate = score_window(
                values[start : start + length],
                canonical_tokens=canonical_tokens,
                collision_tokens=collision_tokens,
                blank_id=blank_id,
                start_frame=start,
            )
            if candidate.collision_margin >= beta:
                candidates.append(candidate)
    if not candidates:
        return WindowScore(0, 0, -math.inf, math.inf, -math.inf)
    return max(candidates, key=lambda item: (item.canonical_fit, item.collision_margin))


def choose_validation_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    negative_exposure_seconds: float,
    min_recall: float = 0.90,
    max_faph: float = 0.10,
) -> dict:
    """Choose the highest-recall qualifying threshold using validation only."""
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("validation requires positive and negative scores")
    exposure_hours = max(float(negative_exposure_seconds) / 3600.0, 1e-12)
    candidates = []
    finite_thresholds = np.unique(
        np.concatenate(
            (positive[np.isfinite(positive)], negative[np.isfinite(negative)])
        )
    )
    for threshold in finite_thresholds:
        recall = float(np.mean(positive >= threshold))
        accepts = int(np.sum(negative >= threshold))
        faph = accepts / exposure_hours
        if recall >= min_recall and faph <= max_faph:
            candidates.append((recall, -faph, float(threshold), accepts))
    required = math.ceil(min_recall * len(positive))
    descending = np.sort(positive)[::-1]
    recall_floor = (
        float(descending[required - 1])
        if required <= len(descending) and math.isfinite(descending[required - 1])
        else None
    )
    floor_accepts = (
        int(np.sum(negative >= recall_floor)) if recall_floor is not None else None
    )
    result = {
        "qualified": bool(candidates),
        "threshold": None,
        "recall": (
            float(np.mean(positive >= recall_floor))
            if recall_floor is not None
            else float(np.mean(np.isfinite(positive)))
        ),
        "faph": (floor_accepts / exposure_hours if floor_accepts is not None else None),
        "false_accepts": floor_accepts,
        "threshold_at_recall_floor": recall_floor,
        "false_accepts_at_recall_floor": floor_accepts,
    }
    if candidates:
        recall, neg_faph, threshold, accepts = max(candidates)
        result.update(
            qualified=True,
            threshold=threshold,
            recall=recall,
            faph=-neg_faph,
            false_accepts=accepts,
        )
    return result


def resolve_phone_ids(tokenizer, phones: Sequence[str]) -> tuple[int, ...]:
    """Resolve IPA phones without relying on AutoProcessor tokenization."""
    ids = tuple(int(tokenizer.convert_tokens_to_ids(phone)) for phone in phones)
    unknown = int(getattr(tokenizer, "unk_token_id", -1))
    if any(token < 0 or token == unknown for token in ids):
        raise ValueError(f"tokenizer cannot represent IPA phone sequence: {phones!r}")
    return ids


def load_hf_teacher(
    model_id: str = MODEL_ID,
    *,
    revision: str = "main",
    device: str = "cpu",
    local_files_only: bool = False,
):
    """Construct the pinned IPA CTC teacher without ``AutoProcessor``."""
    try:
        import torch
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForCTC,
            Wav2Vec2PhonemeCTCTokenizer,
            Wav2Vec2Processor,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "install torch and transformers to load the IPA teacher"
        ) from error
    # A qualified adapted teacher is a complete local Hugging Face directory.
    # ``revision`` is its immutable artifact identity in our reports, not a
    # Hub ref, so never forward it to ``from_pretrained`` for local models.
    local_model = Path(model_id).is_dir()
    common = (
        {"local_files_only": True}
        if local_model
        else {"revision": revision, "local_files_only": local_files_only}
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, **common)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_id, **common)
    processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, **common)
    target = torch.device(device)
    model.to(target).eval()
    return model, processor, tokenizer, target


def resolve_hf_weights_path(
    model_id: str,
    *,
    revision: str,
    local_files_only: bool,
) -> Path:
    """Resolve the exact single-file weights artifact used by a teacher.

    Adaptation deliberately saves one safetensors/bin file so qualification,
    posterior caching, and continuous scoring can all bind the same bytes.
    """
    local = Path(model_id)
    if local.is_dir():
        candidates = tuple(
            path
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path := local / name).is_file()
        )
        if len(candidates) != 1:
            raise ValueError(
                "local teacher must contain exactly one model.safetensors or "
                "pytorch_model.bin weights artifact"
            )
        return candidates[0].resolve()
    from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    for filename in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME):
        resolved = cached_file(
            model_id,
            filename,
            revision=revision,
            local_files_only=local_files_only,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved:
            return Path(resolved).resolve()
    raise ValueError("unable to resolve a single-file teacher weights artifact")


__all__ = [
    "CANONICAL_PHONES",
    "COLLISION_PHONES",
    "MODEL_ID",
    "MODEL_REVISION",
    "TARGET_SAMPLE_RATE",
    "WindowScore",
    "best_window_score",
    "choose_validation_threshold",
    "ctc_fit",
    "ctc_log_probability",
    "ctc_log_probability_batch",
    "load_hf_teacher",
    "resolve_hf_weights_path",
    "resolve_phone_ids",
    "score_window",
    "sha256_file",
    "sha256_text",
]
