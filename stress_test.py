#!./.venv/bin/python3
from NBody.Solver import Solver
import numpy as np

TEST_SIZE = 1e2

test_config = {
    "dtype" : "float64",
    "dt"    : 0.0005,
    "G"     : 1.0,
    "desc"  : "stress test",
    "objs"  : [
        {
            "name": "obj" + str(i),
            "pos": np.random.uniform(-100, 100, 3).tolist(),
            "vel": np.random.uniform(-10, 10, 3).tolist(),
            "mass": np.random.uniform(1, 100),
        }
        for i in range(int(TEST_SIZE))
    ],
    "visuals": {
        "trace_length": 0,
        "fps": 30,
        "ms_per_frame": 20,
        "disp_name": False,
        "scope" : [
            [-100, 100],
            [-100, 100],
            [-100, 100]
        ]
    }
}

solver1 = Solver(config=test_config, visualize=True, precompute=True, title="sim1")

solver1.run(5)