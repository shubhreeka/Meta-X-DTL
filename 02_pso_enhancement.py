"""Step 2: PSO searches enhancement parameters (contrast, brightness, CLAHE clip).

The fitness is computed on a sample of TRAINING images only, so the test set stays unseen.
Output: results/enhancement.json
"""
import json

import numpy as np

from src.config import ARTIFACTS, RESULTS, SEED
from src.enhance import enhance, enhancement_quality
from src.metaheuristics import pso

BOUNDS_LO = [0.7, -40, 1.0]   # alpha (contrast), beta (brightness), CLAHE clip
BOUNDS_HI = [1.8, 40, 4.0]


def main():
    images = np.load(ARTIFACTS / "images.npy")
    split = np.load(ARTIFACTS / "split.npy")
    rng = np.random.default_rng(SEED)
    sample = images[rng.choice(np.where(split == "train")[0], 60, replace=False)]

    baseline = np.mean([enhancement_quality(im) for im in sample])

    def fitness(p):  # minimise negative quality
        return -np.mean([enhancement_quality(enhance(im, *p)) for im in sample])

    best, cost, history = pso(fitness, BOUNDS_LO, BOUNDS_HI, n_particles=12, iters=15, seed=SEED)
    result = {
        "alpha": round(float(best[0]), 4), "beta": round(float(best[1]), 4), "clahe_clip": round(float(best[2]), 4),
        "quality_before": round(float(baseline), 4), "quality_after": round(float(-cost), 4),
        "history": [round(-h, 4) for h in history],
    }
    (RESULTS / "enhancement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
