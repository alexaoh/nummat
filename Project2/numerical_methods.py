"""Implement the numerical methods here, since one needs one neural network per gradient in the formulas."""
from network_algo import *

def symplectic_euler_network(NNT, NNV, q0, p0, times, d0):
    """Symplectic Euler; first order method for integrating functions numerically.
    
    Two trained Neural Networks are input. 
    """

    solution = np.zeros((2*d0, len(times)))
    solution[:, 0] = np.concatenate((q0, p0))

    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1 - t0 # In case the stepsize is not constant.

        grad_T = calculate_gradient(NNT, solution[d0:, n])[:d0,:].reshape(d0)   
        q_new = solution[:d0, n] + stepsize*grad_T # Reshape is necessary for dimension.

        grad_V = calculate_gradient(NNV, q_new)[:d0,:].reshape(d0)
        p_new = solution[d0:, n] - stepsize*grad_V

        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new
    return solution

def symplectic_euler_exact(q0, p0, times, grad_T, grad_V, d0):
    """Symplectic Euler; first order method for integrating functions numerically.
    
    The exact gradients (functions) of V and T in the problems are used. 
    """

    solution = np.zeros((2*d0,len(times)))
    solution[:, 0] = np.concatenate((q0, p0))
    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1 - t0

        q_new = solution[:d0,n] + stepsize*grad_T(solution[d0:,n])
        p_new = solution[d0:,n] - stepsize*grad_V(q_new)
    
        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new

    return solution

def stormer_verlet_network(NNT, NNV, q0, p0, times, d0):
    """Størmer-Verlet; second order method for integrating functions numerically.

    Two trained Neural Networks are input.
    """
    solution = np.zeros((2*d0, len(times)))
    solution[:, 0] = np.concatenate((q0, p0))

    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1 - t0 # In case the stepsize is not constant.

        grad_V_half = calculate_gradient(NNV, solution[:d0, n])[:d0,:].reshape(d0)
        p_half = solution[d0:,n] - stepsize/2 * grad_V_half

        grad_T = calculate_gradient(NNT, p_half)[:d0,:].reshape(d0)
        q_new = solution[:d0,n] + stepsize * grad_T

        grad_V = calculate_gradient(NNV, q_new)[:d0,:].reshape(d0)
        p_new = p_half - stepsize/2 * grad_V

        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new

    return solution

def stormer_verlet_exact(q0, p0, times, grad_T, grad_V, d0):
    """Størmer-Verlet; second order method for integrating functions numerically.
    
    The exact gradients of V and T in the problems are used. 
    """
    solution = np.zeros((2*d0,len(times)))
    solution[:, 0] = np.concatenate((q0, p0))
    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1 -t0

        p_half = solution[d0:,n] - stepsize/2 * grad_V(solution[:d0,n])
        q_new = solution[:d0,n] + stepsize * grad_T(p_half)
        p_new = p_half - stepsize/2 * grad_V(q_new)

        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new

    return solution

def calculate_gradient(NN, point):
    """Calculate gradient in specific point from network."""
    # Embed new data. 
    NN.embed_input(point)

    # Run forward function.
    NN.forward_function()

    # Calculate gradient.
    grad = NN.Hamiltonian_gradient()
    return grad


