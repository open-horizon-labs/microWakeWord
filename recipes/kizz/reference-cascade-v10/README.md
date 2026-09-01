# Kizz Control cascade v10 reference artifacts

These are the exact three INT8 TensorFlow Lite models in the v10 StackChan
firmware. V10 keeps the frozen detector and ordered verifier, and replaces the
middle gate with a model trained from correctly captured full-pre-roll device
audio. The exact firmware passed startup AOT/reference equivalence and 12/12
held-out physical speaker replays on an ESP32-S3 revision 0.2.

| Role | File | SHA-256 |
| --- | --- | --- |
| Permissive ordered detector | `kizz_control_detector.tflite` | `f07d2c010fba020e923c23734e54ba8e86751dfd1b0f23a018eb5ff79b969ae3` |
| Compact full-pre-roll verifier | `kizz_control_candidate_verifier_int8.tflite` | `a28e8c8f3fe51ea3ae3fc76f0d79f2abdb06f19c71d7f26e0f08a16464025710` |
| Independent ordered verifier | `kizz_control_ordered_verifier_int8.tflite` | `956a444d11f802e7780dcd3af6f43551a1fe4601fdacfd7b153bba8e11c48933` |

The adjacent metadata and threshold reports retain the conversion, operator,
quantization, split, physical-replay, and selection contracts. Some immutable
reports include provenance paths from the producing machine; resolve the
checked-in files by filename and SHA-256, never by those historical paths.

Active thresholds:

- detector ordered score: `-18.20059454471544`;
- compact verifier logit: `0.0`;
- ordered verifier score: `-19.326665980378795`.

The compact metadata deliberately names
`kizz_control_candidate_verifier_int8.tflite`; the bundled filename matches so
the evaluator's exact artifact binding works in a fresh checkout.

See [the full recipe](../CASCADE_V10_RECIPE.md) for the evidence, AOT/ESP-NN
implementation, and remaining soak boundary.
