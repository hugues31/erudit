from erudit.utils import decay_schedule


def test_decay_schedule():
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    values = decay_schedule(0, 1, 1, 5)
    expected = np.array([0.0, 0.69067902, 0.90909091, 0.97815881, 1.0])
    assert np.isclose(values, expected).all()
