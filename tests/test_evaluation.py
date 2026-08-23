import numpy as np
import pytest

from microwakeword.test import false_negative_rates_at_cutoffs


def test_false_negative_rates_reject_split_without_positives():
    with pytest.raises(
        ValueError,
        match="testing must contain at least one positive sample to compute ROC",
    ):
        false_negative_rates_at_cutoffs([], np.asarray([0.5]), "testing")


def test_false_negative_rates_use_strict_operating_cutoff():
    rates = false_negative_rates_at_cutoffs(
        [0.5, 0.75], np.asarray([0.5, 0.7, 0.8]), "testing"
    )
    np.testing.assert_allclose(rates, [0.5, 0.5, 1.0])
