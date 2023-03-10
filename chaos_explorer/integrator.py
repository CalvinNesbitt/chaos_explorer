# Imports
from scipy.integrate import solve_ivp


class OdeIntegrator:
    """
    Integrates a determinsitic dynamical system.
    """

    def __init__(self, rhs, ic, parameters={}, method="RK45"):
        """
        rhs, function: Maps from state to rhs of ode.
        parameters, dict: Parameters used in ode.
        ic, np.array: initial condition for the ode.
        method, string: passed to solve_ivp.
        """
        self.rhs = rhs
        self.ic = ic
        self.state = ic
        self.parameters = parameters
        self.time = 0
        self.method = method
        self.ndim = len(ic)

    def _rhs_dt(self, t, state):
        return self.rhs(state, **self.parameters)

    def run(self, t):
        """t: how long we integrate for in adimensional time."""

        # Integration, default uses RK45 with adaptive stepping.
        solver_return = solve_ivp(
            self._rhs_dt,
            (self.time, self.time + t),
            self.state,
            dense_output=True,
            method=self.method,
        )

        # Updating variables
        self.state = solver_return.y[:, -1]
        self.time = self.time + t
