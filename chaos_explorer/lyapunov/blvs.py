# TODO: Wrapper for the computation of BLVs

from .core import BennetinIntegrator

import numpy as np
from loguru import logger
import sys


def compute_blvs(rhs, jacobian, number_of_observations, ic, tau=0.1, transient_len=1000):
    bennetin_stepper = BennetinIntegrator(rhs, ic, jacobian, np.eye(3) * 1.0e-6, tau=tau)

    logger.remove()
    logger.add(sys.stdout, colorize=False, format="{time:YYYYMMDDHHmmss}|{level}|{message}")

    # Run Transient
    bennetin_stepper.many_steps(transient_len)
    bennetin_stepper.time = 0

    # Initialise observations
    ftble_ts = []
    blv_ts = []
    trajectory_ts = []

    for step in range(int(number_of_observations)):
        # Do Bennetin Steps
        bennetin_stepper.step()

        # Compute FTBLE/BLV
        ftbles = np.log(np.diag(bennetin_stepper.R)) / bennetin_stepper.tau
        blvs = bennetin_stepper.Q

        # Observe
        ftble_ts.append(ftbles)
        blv_ts.append(blvs)
        trajectory_ts.append(bennetin_stepper._trajectory_state)

    return trajectory_ts, blv_ts, ftble_ts
