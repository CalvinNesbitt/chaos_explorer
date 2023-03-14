import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
import sys

from chaos_explorer.tangent_integrator import TangentIntegrator


def posQR(M):
    """Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M)  # Performing QR decomposition
    signs = np.diag(np.sign(np.diagonal(R)))  # Matrix with signs of R diagonal on the diagonal
    Q, R = np.dot(Q, signs), np.dot(signs, R)  # Ensuring R Diagonal is positive
    return Q, R


class BennetinStepper(TangentIntegrator):
    "Performs the Bennetin steps."

    def __init__(self, rhs, ic, jacobian, Q_ic, tau=0.01, parameters={}, method="RK45"):
        super().__init__(rhs, ic, jacobian, perturbation_ic=Q_ic[0], parameters=parameters, method=method)

        self.Q = Q_ic
        self.R = np.zeros(Q_ic.shape)
        self.tau = tau

    def step(self):
        # Trajectory state prior to pusing matrix forward
        time = self.time
        trajectory_state = self._trajectory_state
        P = np.zeros(self.Q.shape)  # dummy stretched matrix

        # Integrate perturbation matrix forward
        for i, column in enumerate(self.Q.T):
            self.time = time
            self._trajectory_state = trajectory_state
            self._perturbation_state = column
            self.run(self.tau)  # use underlying tangent integrator
            P.T[i] = self._perturbation_state

        # Updata Q and R
        self.Q, self.R = posQR(P)
        return

    def many_steps(self, n):
        for i in range(n):
            self.step()


logger.remove()
logger.add(sys.stdout, colorize=False, format="{time:YYYYMMDDHHmmss}|{level}|{message}")


class BennetinObserver:
    def __init__(self, bennetin_stepper, quiet=False):
        self.bennetin_stepper = bennetin_stepper
        self._time_obs = []
        self.dump_count = 0
        self._Q_observations = []
        self._R_observations = []
        self._trajectory_observations = []
        self.store_Q = True
        self.store_R = True
        self.quiet = quiet

    def make_observations(self, number, timer=True):
        self.look(self.bennetin_stepper)  # Initial observation
        for x in tqdm(range(number), disable=not timer):
            self.bennetin_stepper.step()
            self.look(self.bennetin_stepper)
        return

    def make_observations_in_blocks(self, save_folder, number_of_obs, block_size, timer=True):
        number_of_blocks = int(number_of_obs / block_size)
        remainder = int(number_of_obs % block_size)
        self.look(self.bennetin_stepper)  # Initial observation
        for block in tqdm(range(number_of_blocks), disable=not timer):
            for i in range(block_size):
                self.bennetin_stepper.step()
                self.look(self.bennetin_stepper)
            self.dump(save_folder / f"{block}.nc")

        if remainder != 0:
            for i in range(remainder):
                self.bennetin_stepper.step()
                self.look(self.bennetin_stepper)
            self.dump(save_folder / f"{block + 1}.nc")
        return

    def look(self, bennetin_stepper):
        self._time_obs.append(bennetin_stepper.time)
        if self.store_Q:
            self._Q_observations.append(bennetin_stepper.Q)
        if self.store_R:
            self._R_observations.append(bennetin_stepper.R)
        self._trajectory_observations.append(bennetin_stepper._trajectory_state)

    @property
    def observations(self):
        if len(self._R_observations) == 0:
            print("I have no observations! :(")
            return

        # Set up dimensions
        le_index = np.arange(1, 1 + self.bennetin_stepper.ndim)
        component = np.arange(1, 1 + self.bennetin_stepper.ndim)
        time = self._time_obs

        # Package Observations
        dic = {}
        if self.store_Q:
            dic["Q"] = xr.DataArray(
                self._Q_observations,
                dims=["time", "le_index", "component"],
                name="Q",
                coords={"time": time, "le_index": le_index, "component": component},
            )
        if self.store_R:
            dic["R"] = xr.DataArray(
                self._R_observations,
                dims=["time", "le_index", "component"],
                name="R",
                coords={"time": time, "le_index": le_index, "component": component},
            )
        dic["trajectory"] = xr.DataArray(
            self._trajectory_observations,
            dims=["time", "component"],
            name="trajectory",
            coords={"time": time, "component": component},
        )
        return xr.Dataset(dic)

    def wipe(self):
        """Erases observations"""
        self._time_obs = []
        self._R_observations = []
        self._Q_observations = []
        self._trajectory_observations = []

    def dump(self, save_name):
        """Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if len(self._R_observations) == 0:
            print("I have no observations! :(")
            return

        self.observations.to_netcdf(save_name)
        if not self.quiet:
            logger.info(f"Observations written to {save_name}. Erasing personal log.\n")
        self.wipe()
        self.dump_count += 1
        return


# def clv_convergence_step(R, A):
#     newA = np.linalg.solve(R, A)  # Push A with R^-1
#     norms = np.linalg.norm(newA, axis=0, ord=2)  # Prevent vector growth
#     return newA / norms


# def many_clv_convergence_steps(R_ts, A):
#     A_ts = []
#     for R in R_ts:
#         A_ts.append(A)
#         A = clv_convergence_step(R, A)
#     return A_ts
