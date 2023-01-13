# Imports
from scipy.integrate import solve_ivp


class odeIntegrator:
    """
    Integrates a determinsitic dynamical system.
    """

    def __init__(self, rhs, ic, parameters={}):
        """
        rhs, function: Maps from state to rhs of ode.
        parameters, dict: Parameters used in ode.
        ic, np.array: initial condition for the ode.
        """
        self.rhs = rhs
        self.state = ic
        self.parameters = parameters

    def _rhs_dt(self, t, state):
        return self.rhs(state, **self.parameters)

    def integrate(self, t):
        """t: how long we integrate for in adimensional time."""

        # Integration, uses RK45 with adaptive stepping.
        solver_return = solve_ivp(
            self._rhs_dt, (self.time, self.time + t), self.state, dense_output=True
        )

        # Updating variables
        self.state = solver_return.y[:, -1]
        self.time = self.time + t
