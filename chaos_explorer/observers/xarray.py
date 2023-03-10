from .base import Observer
from tqdm import tqdm
import xarray as xr


class XarrayObserver(Observer):
    """Parent class that has the basic functionality we expect from an xarray observer.
    Child classes should be equipped with a method called 'observations'
    that unpacks the ._observations list into xr."""

    def __init__(self, integrator):
        """param, integrator: integrator being observed."""

        # Needed knowledge of the integrator
        self.parameters = integrator.parameters
        self.integrator = integrator

        # Observation logs
        self._time_obs = []  # Times we've made observations
        self._observations = []
        self.dump_count = 0

    def make_observations(self, number, frequency, timer=True):
        self.look(self.integrator)  # Initial observation
        for x in tqdm(range(number), disable=not timer):
            self.integrator.run(frequency)
            self.look(self.integrator)
        return

    def wipe(self):
        """Erases observations"""
        self._time_obs = []
        self._observations = []

    def dump(self, save_name):
        """Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if len(self._observations) == 0:
            print("I have no observations! :(")
            return

        self.observations.to_netcdf(save_name)
        print(f"Observations written to {save_name}. Erasing personal log.\n")
        self.wipe()
        self.dump_count += 1
        return


class TrajectoryObserver(XarrayObserver):
    def look(self, integrator):
        """Observes trajectory of a integrator"""

        # Note the time
        self._time_obs.append(integrator.time)

        # Making Observations
        self._observations.append(integrator.state.copy())
        return


class TangentTrajectoryObserver(XarrayObserver):
    def look(self, tangent_integrator):
        """Observes trajectory of a tangent integrator"""

        # Note the time
        self._time_obs.append(tangent_integrator.time)

        # Making Observations
        self._observations.append(tangent_integrator._trajectory_state.copy())
        return


class ScalarObserver(XarrayObserver):
    def __init__(self, integrator, scalar_function, name: str):
        super().__init__(integrator)
        self.name = name
        self.scalar_function = scalar_function

    def look(self, integrator):
        """Observes scalar valur of trajectory"""

        # Note the time
        self._time_obs.append(integrator.time)

        # Making Observations
        self._observations.append(self.scalar_function(integrator.state.copy()))
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if len(self._observations) == 0:
            print("I have no observations! :(")
            return

        dic = {}
        _time = self._time_obs
        xr_da = xr.DataArray(
            self._observations,
            dims=["time"],
            name=self.name,
            coords={"time": _time},
            attrs=self.parameters,
        )
        return xr_da


class MstateObserver(XarrayObserver):
    "Observes and Mstate computation."

    def make_observations(self, number, timer=True):
        self.look(self.integrator)  # Initial observation
        for x in tqdm(range(number), disable=not timer):
            self.integrator.run(1)  # run here is # of mstate alg steps rather than length of time
            self.look(self.integrator)
        return

    def look(self, mstate_alg):
        """Observes"""

        # Note the time
        self._time_obs.append(mstate_alg.time)

        # Making Observations
        self._observations.append(mstate_alg.midpoint.copy())
        return
