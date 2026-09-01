# Kizz Control cascade v9 reference artifacts

These are the exact three INT8 TensorFlow Lite models used by the accepted v9
host cascade result. They are reference and firmware-handoff artifacts, not a
claim of complete StackChan hardware qualification.

| Role | File | SHA-256 |
| --- | --- | --- |
| Permissive ordered detector | `kizz_control_detector.tflite` | `f07d2c010fba020e923c23734e54ba8e86751dfd1b0f23a018eb5ff79b969ae3` |
| Compact device-adapted verifier | `kizz_control_compact_verifier_int8_v9.tflite` | `0265dc7e56823f0bb774c641d9e1f01637f075adc60ae6e110c8786e08a622f8` |
| Independent ordered verifier | `kizz_control_ordered_verifier_int8.tflite` | `956a444d11f802e7780dcd3af6f43551a1fe4601fdacfd7b153bba8e11c48933` |

The adjacent metadata and threshold reports retain the conversion, operator,
quantization, and selection contracts. Some historical provenance paths inside
those immutable reports point to the machine that produced the artifacts; use
the hashes, not those paths, to resolve the checked-in files.

Active thresholds:

- detector ordered score: `-18.20059454471544`;
- compact verifier logit: `0.0`;
- ordered verifier score: `-19.326665980378795`.

See [the full recipe](../CASCADE_V9_RECIPE.md) for the evidence and remaining
physical qualification boundary.
