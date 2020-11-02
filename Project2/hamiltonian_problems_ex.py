"""The supplied examples of separable Hamiltonian problems."""
import numpy as np

class Pendulum:
    """Nonlinear pendulum."""
    def __init__(self, domain_T, domain_V):
        self.domain_T = domain_T
        self.domain_V = domain_V
        self.k = 10 # Constant for m*g*l (e.g).

    def T(self, p):
        return 0.5*p**2

    def grad_T(self, p):
        return p

    def V(self, q):
        return self.k*(1-np.cos(q))

    def grad_V(self, q):
        return self.k*np.sin(q)

class Kepler:
    """Kepler two-body problem."""
    def __init__(self, domain_T, domain_V):
        self.domain_T = domain_T
        self.domain_V = domain_V
        
    def T(self, p):
        return 0.5*(p[0]**2 + p[1]**2)

    def grad_T(self, p):
        return np.array([p[0], p[1]])

    def V(self, q):
        return -1/np.sqrt(q[0]**2 + q[1]**2)
    
    def grad_V(self, q):
        comp1 = q[0]/np.power(q[0]**2 + q[1]**2, 3/2)
        comp2 = q[1]/np.power(q[0]**2 + q[1]**2, 3/2)
        return np.array([comp1, comp2]).T

class Henon_Heiles:
    """Henon-Heiles problem."""
    def __init__(self, domain_T, domain_V):
        self.domain_T = domain_T
        self.domain_V = domain_V
        
    def T(self, p):
        return 0.5*(p[0]**2 + p[1]**2)
    
    def grad_T(self, p):
        return np.array([p[0], p[1]])

    def V(self, q):
        return 0.5*(q[0]**2 + q[1]**2) + q[0]**2*q[1] - (1/3)*q[1]**3
    
    def grad_V(self, q):
        comp1 = q[0] + 2*q[0]*q[1]
        comp2 = q[1] + q[0]**2 - q[1]**2

        return np.array([comp1,comp2])
