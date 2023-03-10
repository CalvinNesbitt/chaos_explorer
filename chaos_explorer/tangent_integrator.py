import numpy as np
from scipy.integrate import solve_ivp


class TangentIntegrator:
    def __init__(self, rhs, ic, jacobian, perturbation_ic=None, parameters={}, method="RK45"):

        self.rhs = rhs
        self.jacobian = jacobian
        self._trajectory_state = ic
        self.ndim = len(ic)
        self._perturbation_state = perturbation_ic
        self.time = 0
        self.method = method
        self.parameters = parameters

    @property
    def state(self):
        return np.append(self._trajectory_state, self._perturbation_state)

    def _trajectory_rhs_dt(self, trajectory):
        return self.rhs(trajectory, **self.parameters)

    def _tangent_rhs_dt(self, trajectory, perturbation):
        return self.jacobian(trajectory, **self.parameters).dot(perturbation)

    def _tlm_rhs_dt(self, t, state):
        trajectory = state[: self.ndim]
        perturbation = state[self.ndim :]
        trajectory_rhs = self._trajectory_rhs_dt(trajectory)
        tangent_rhs_dt = self._tangent_rhs_dt(trajectory, perturbation)
        return np.append(trajectory_rhs, tangent_rhs_dt)

    def run(self, t):
        """t: how long we integrate for in adimensional time."""

        # Integration, default uses RK45 with adaptive stepping.
        solver_return = solve_ivp(
            self._tlm_rhs_dt,
            (self.time, self.time + t),
            self.state,
            dense_output=True,
            method=self.method,
        )

        # Updating variables
        self._trajectory_state = solver_return.y[: self.ndim, -1]
        self._perturbation_state = solver_return.y[self.ndim :, -1]
        self.time = self.time + t


# TODO: Implement auto jacobian using jax.jaxfwd
