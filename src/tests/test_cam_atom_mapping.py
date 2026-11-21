import numpy as np

from model_explanation.cam_atom_mapping import cam_to_atom_scores


def test_cam_to_atom_scores_matches_rdkit_atom_count():
    yy, xx = np.mgrid[0:256, 0:256]
    heatmap = ((xx + yy) / 510.0).astype(np.float32)
    scores = cam_to_atom_scores("CCO", heatmap, radius=8, aggregation="mean")
    assert list(scores) == [0, 1, 2]
    assert all(0.0 <= value <= 1.0 for value in scores.values())
