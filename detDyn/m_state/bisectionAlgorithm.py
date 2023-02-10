"""
Class defining M State bisection algorithm.
To implement you need to write integrator, check_cold, cool_down, heat_up functions.
Example of each is commented in this doc.
"""
from tqdm import tqdm


def bisect(x, y):
    return x + 0.5 * (y - x)


class BisectionAlgorithm:
    """
    Calculates the melancholia state of a bistable dynamical system.

    Inputs
    ------------
    integrator, object
        Something that runs the dynamical system.
        Needs to have set_state, state and run functions.

    check_cold, function
        Function that tests whether a point is in the 'cold' basin.
        Returns True or False

    cool_down, function
        Function that pushes state towards cold basin.

    heat_up, functiomn
        Function that pushes state towards hot basin.

    tau, float
        The time between bisections.

    ic, list
        Where to start algorithm from [cold basin ic, hot basin ic].
    """

    def __init__(self, integrator, check_cold, cool_down, heat_up, tau, ic):

        self.integrator = integrator
        self.parameters = integrator.parameters
        self.check_cold = check_cold
        self.cool_down = cool_down
        self.heat_up = heat_up
        self.cold_point, self.hot_point = ic
        self.midpoint = bisect(self.cold_point, self.hot_point)
        self.tau = tau
        self.time = 0

    def _midpoint_update(self):
        "Calculates midpoint and updates hot/cold points respectively"

        # Find midpoint
        self.midpoint = bisect(self.cold_point, self.hot_point)

        # Check if it the midpoint is cold or hot
        midpoint_cold = self.check_cold(self.midpoint, self.integrator)

        # Update cold or hot point depending on result
        if midpoint_cold:
            self.cold_point = self.midpoint
            self.heat_up(self.hot_point)  # Keep
        elif not midpoint_cold:
            self.hot_point = self.midpoint
            self.cool_down(self.cold_point)
        elif midpoint_cold is None:
            self.cold_point = self.midpoint

    def _step(self):
        "One step on M-state algorithm"

        # Integate cold and hot points one step forward
        self.integrator.state = self.cold_point
        self.integrator.run(self.tau)
        self.cold_point = self.integrator.state

        self.integrator.state = self.hot_point
        self.integrator.run(self.tau)
        self.hot_point = self.integrator.state

        # Midpoint update
        self._midpoint_update()
        self.time += self.tau

    def run(self, steps, timer=False):
        "Many step of m-state algorithm"
        for i in tqdm(range(steps), disable=(not timer)):
            self._step()


#
# Example check_cold, heat_up and cool_down functions
#
# def check_cold(ic, integrator):
#     """
#     Checks whether a given ic ends up at the cold point.
#     """

#     integrator.set_state(ic)
#     tau = 0.1 # How long we run between checks, will effect how efficient we are

#     for i in range(1000): # How many checks we make
#         integrator.run(tau)
#         if integrator.state[0] < -0.5: #Threshold for being cold, ensure cold ic matches this
#             return True
#         elif integrator.state[0] > 0.5: #Threshold for being hot
#             return False
#     return None

# def heat_up(x):
#     x[-1] += 0.01

# def cool_down(x):
#     x[-1] -= 0.01
