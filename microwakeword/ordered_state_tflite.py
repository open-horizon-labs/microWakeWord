"""TensorFlow Lite output adapter for the ordered-state logit decoder."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from microwakeword.ordered_state import KIZZ_TOPOLOGY, OrderedStateTopology


def tflite_output_logits(
    output,
    output_detail: Mapping[str, Any],
    topology: OrderedStateTopology = KIZZ_TOPOLOGY,
) -> np.ndarray:
    """Dequantize one TFLite output tensor into a state-logit vector."""
    values = np.asarray(output)
    if values.size != topology.state_count:
        raise ValueError(
            f"expected one {topology.state_count}-state output, got shape {values.shape}"
        )
    if np.issubdtype(values.dtype, np.integer):
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("quantized ordered-state output needs a positive scale")
        values = (values.astype(np.float32) - float(zero_point)) * float(scale)
    elif not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"unsupported ordered-state output dtype: {values.dtype}")
    values = values.astype(np.float32, copy=False).reshape(topology.state_count)
    if np.any(~np.isfinite(values)):
        raise ValueError("ordered-state output contains non-finite logits")
    return values
