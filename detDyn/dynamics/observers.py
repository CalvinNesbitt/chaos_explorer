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
