import numpy as np
from chaos_explorer.tangent_integrator import TangentIntegrator


def posQR(M):
    """Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M)  # Performing QR decomposition
    signs = np.diag(np.sign(np.diagonal(R)))  # Matrix with signs of R diagonal on the diagonal
    Q, R = np.dot(Q, signs), np.dot(signs, R)  # Ensuring R Diagonal is positive
    return Q, R


class BennetinIntegrator(TangentIntegrator):
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
