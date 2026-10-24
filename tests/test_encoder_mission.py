import numpy as np

from robotlab.cases import get_case
from robotlab.simulation import run


def test_encoder_outage_stops_then_resumes():
    result = run(get_case("encoder-outage"))
    journal = result["journal"]
    onset = next(e for e in journal if e["reason"] == "encoder_timeout")
    recovery = next(e for e in journal if e["previous"] == "encoder_timeout")
    assert 1.1 < onset["time"] < 1.3
    assert 2 <= recovery["time"] < 2.1
    time = result["time"]
    assert np.allclose(result["command"][(time > 1.3) & (time < 2)], 0)
    assert np.max(result["state"][(time > 2.5), 4]) > 0.1
    assert result["clearance"].min() > 0
