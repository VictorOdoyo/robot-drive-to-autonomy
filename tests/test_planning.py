import numpy as np
import pytest

from robotlab import planning as m


def test_adjacency_blocks_corner_cutting():
    blocked = np.array([[False, True], [True, False]])
    assert list(m.neighbors((0, 0), blocked)) == []
    assert m.octile((0, 0), (3, 2)) == pytest.approx(1 + 2 * np.sqrt(2))
