class baseObserver:

    """Parent class that has the basic functionality we expect from an observer.
    Child classes should be equipped with a property called 'observations'
    that unpacks the ._observations list into xr."""

    def __init__(self, integrator, name=""):
        """param, integrator: integrator being observed."""

        # Needed knowledge of the integrator
        self._parameters = integrator.parameters

        # Observation logs
        self.time_obs = []  # Times we've made observations
        self._observations = []
        self.dump_count = 0

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
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


class trajectoryObserver(baseObserver):
    """Observes the trajectory of Asymettric Double Well. Dumps to netcdf."""

    def __init__(self, integrator, name=''):
        """param, integrator: integrator being observed."""
        super().__init__(integrator, name=name)

    def look(self, integrator):
        """Observes trajectory """

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.x_obs.append(integrator.state[0].copy())
        self.y_obs.append(integrator.state[1].copy())
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['X'] = xr.DataArray(self.x_obs, dims=['time'], name='X',
                                coords = {'time': _time})
        dic['Y'] = xr.DataArray(self.y_obs, dims=['time'], name='Y',
                                coords = {'time': _time})
        return xr.Dataset(dic, attrs= self._parameters)