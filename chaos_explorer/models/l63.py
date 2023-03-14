from chaos_explorer.observers.xarray import TrajectoryObserver

import numpy as np
import xarray as xr


def l63(state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def l63_jacobian(state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    grad_f0 = np.array([-sigma, sigma, 0])
    grad_f1 = np.array([-z + rho, -1, -x])
    grad_f2 = np.array([y, x, -beta])
    return np.array([grad_f0, grad_f1, grad_f2])


class L63TrajectoryObserver(TrajectoryObserver):
    @property
    def observations(self):
        if len(self._observations) == 0:
            print("I have no observations! :(")
            return

        dic = {}
        dic["X"] = xr.DataArray(
            np.stack(self._observations)[:, 0],
            dims=["time"],
            name="X",
            coords={"time": self._time_obs},
        )

        dic["Y"] = xr.DataArray(
            np.stack(self._observations)[:, 1],
            dims=["time"],
            name="X",
            coords={"time": self._time_obs},
        )

        dic["Z"] = xr.DataArray(
            np.stack(self._observations)[:, 2],
            dims=["time"],
            name="X",
            coords={"time": self._time_obs},
        )
        return xr.Dataset(dic, attrs=self.parameters)
