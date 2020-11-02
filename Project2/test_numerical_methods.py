"""Test the numerical methods on the supplied examples of separable Hamiltonian problems."""
from numerical_methods import *
from import_data import *
from hamiltonian_problems_ex import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#====================#
# Nonlinear Pendulum #
#====================#

def nonlinear_pendulum_test():
    """Train networks, calculate numerical solutions and plot figures for the nonlinear pendulum problem."""
    domain_T = domain_V = [-1, 1]
    pendulum = Pendulum(domain_T, domain_V)
    d0 = 1
    I = 1000
    d = 4
    K = 30
    h = 0.1
    iterations = 5000
    tau = None
    chunk = int(I/10)

    # Train the neural networks. 
    NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.T, pendulum.domain_T)
    NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.V, pendulum.domain_V)

    # Initial position and momentum.
    q0 = np.array([0.2])
    p0 = np.array([0.2])
    timesteps = 1000
    times = np.linspace(0, 10, timesteps)

    # Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
    network_sol_euler = symplectic_euler_network(NNT, NNV, q0, p0, times,d0)
    exact_sol_euler = symplectic_euler_exact(q0, p0, times, pendulum.grad_T, pendulum.grad_V, d0)
    network_sol_SV = stormer_verlet_network(NNT, NNV, q0, p0, times,d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, pendulum.grad_T, pendulum.grad_V, d0)

    # Plot the exact position and the network position resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_sol_euler[0,:], linewidth = 0.5, label = "Network (Euler)")
    plt.plot(times, exact_sol_euler[0,:], linewidth = 0.5, label = "Exact (Euler)")
    plt.plot(times, network_sol_SV[0,:], linewidth = 0.5, label = "Network (SV)")
    plt.plot(times, exact_sol_SV[0,:], linewidth = 0.5, label = "Exact (SV)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("q")
    fig.suptitle("Position against Time", fontweight = "bold")
    plt.savefig("PendulumPosTime.pdf", bbox_inches='tight')
    plt.show()

    # Plot the exact momentum and the network momentum resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_sol_euler[1,:], linewidth = 0.5, label = "Network (Euler)")
    plt.plot(times, exact_sol_euler[1,:], linewidth = 0.5, label = "Exact (Euler)")
    plt.plot(times, network_sol_SV[1,:], linewidth = 0.5, label = "Network (SV)")
    plt.plot(times, exact_sol_SV[1,:], linewidth = 0.5, label = "Exact (SV)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("p")
    fig.suptitle("Momentum against Time", fontweight = "bold")
    plt.savefig("PendulumMomentumTime.pdf", bbox_inches='tight')
    plt.show()

    network_ham_euler = NNT.calculate_output(network_sol_euler[1,:].reshape(1,timesteps)) + NNV.calculate_output(network_sol_euler[0,:].reshape(1,timesteps))
    network_ham_SV = NNT.calculate_output(network_sol_SV[1,:].reshape(1,timesteps)) + NNV.calculate_output(network_sol_SV[0,:].reshape(1,timesteps))
    exact_ham_euler = pendulum.T(exact_sol_euler[1,:])+pendulum.V(exact_sol_euler[0,:])
    exact_ham_SV = pendulum.T(exact_sol_SV[1,:])+pendulum.V(exact_sol_SV[0,:])

    # Plot Hamiltonian with exact gradient. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, exact_ham_euler, label = "Exact (Euler)", color = "red", linestyle = "dashed")
    exact_ham_euler = pendulum.T(exact_sol_euler[1,:])+pendulum.V(exact_sol_euler[0,:])
    plt.plot(times, exact_ham_SV, color = "blue", label = "Exact (SV)")
    plt.axhline(y = pendulum.T(p0)+pendulum.V(q0), color = "yellow", label = "Correct")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("PendulumHamiltonianExact.pdf", bbox_inches='tight')
    plt.show()

    # Plot Hamiltonian with approximate gradient from network.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_ham_euler, label = "Network (Euler)", color = "red", linestyle = "dashed")
    plt.plot(times, network_ham_SV, color = "blue", label = "Network (SV)")
    plt.axhline(y = pendulum.T(p0)+pendulum.V(q0), color = "yellow", label = "Correct")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("PendulumHamiltonianNetwork.pdf", bbox_inches='tight')
    plt.show()

#=========================#
# Kepler Two-body Problem #
#=========================#

def kepler_test():
    """Train networks, calculate numerical solutions and plot figures for the Kepler Two-body problem."""
    domain_T = domain_V = [[-2, 2], [-2, 2]]
    kepler = Kepler(domain_T, domain_V)
    d0 = 2
    I = 1000
    d = 4
    K = 30
    h = 0.1
    iterations = 5000
    tau = 0.01
    method = "adams"
    chunk = int(I/10)

    # Train the neural networks. 
    NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, kepler.T, kepler.domain_T)
    NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, kepler.V, kepler.domain_V)

    e = 0.00001 # e is between 0 and 1.
    q0  = np.array([1-e,0])
    p0 = np.array([0,np.sqrt((1+e)/1-e)])
    timesteps = 1000
    times = np.linspace(0, 40, timesteps)

    # Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
    network_sol_euler = symplectic_euler_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_euler = symplectic_euler_exact(q0, p0, times, kepler.grad_T, kepler.grad_V, d0)
    network_sol_SV = stormer_verlet_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, kepler.grad_T, kepler.grad_V, d0)

    # Plot the exact position and the network position resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times 
    xline = exact_sol_SV[0,:]
    yline = exact_sol_SV[1,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_SV[0,:]
    ynet = network_sol_SV[1,:]
    ax.plot3D(xnet, ynet, zline, "blue", label = "Network (SV)") 
    xline = exact_sol_euler[0,:]
    yline = exact_sol_euler[1,:]
    ax.plot3D(xline, yline, zline, "black", label = "Exact (Euler)", linestyle = "dashed")
    xnet = network_sol_euler[0,:]
    ynet = network_sol_euler[1,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)", linestyle = "dashed") 
    plt.legend()
    fig.suptitle("Position in Time Space", fontweight = "bold")
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("KeplerPosition.pdf", bbox_inches='tight')
    plt.show()

    # Plot the exact momentum and the network momentum resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times 
    xline = exact_sol_SV[2,:]
    yline = exact_sol_SV[3,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_SV[2,:]
    ynet = network_sol_SV[3,:]
    ax.plot3D(xnet, ynet, zline, "blue", label = "Network (SV)") 
    xline = exact_sol_euler[2,:]
    yline = exact_sol_euler[3,:]
    ax.plot3D(xline, yline, zline, "black", label = "Exact (Euler)", linestyle = "dashed")
    xnet = network_sol_euler[2,:]
    ynet = network_sol_euler[3,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)", linestyle = "dashed") 
    plt.legend()
    fig.suptitle("Momentum in Time Space", fontweight = "bold")
    ax.set_xlabel("$p_1$")
    ax.set_ylabel("$p_2$")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("KeplerMomentum.pdf", bbox_inches='tight')
    plt.show()

    network_ham_euler = NNT.calculate_output(network_sol_euler[d0:,:].reshape(d0,timesteps)) + NNV.calculate_output(network_sol_euler[:d0,:].reshape(d0,timesteps))
    network_ham_SV = NNT.calculate_output(network_sol_SV[d0:,:].reshape(d0,timesteps)) + NNV.calculate_output(network_sol_SV[:d0,:].reshape(d0,timesteps))
    exact_ham_euler = kepler.T(exact_sol_euler[d0:,:])+kepler.V(exact_sol_euler[:d0,:])
    exact_ham_SV = kepler.T(exact_sol_SV[d0:,:])+kepler.V(exact_sol_SV[:d0,:])

    # Plot Hamiltonian with exact gradient. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, exact_ham_euler, label = "Exact (Euler)", color = "red", linestyle = "dashed")
    plt.axhline(y = kepler.T(p0)+kepler.V(q0), color = "yellow", label = "Correct")
    plt.plot(times, exact_ham_SV, color = "blue", label = "Exact (SV)", linestyle = "dashed")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("KeplerHamiltonianExact.pdf", bbox_inches='tight')
    plt.show()

    # Plot Hamiltonian with approximate gradient from network.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_ham_euler, label = "Network (Euler)", color = "red")
    plt.axhline(y = kepler.T(p0)+kepler.V(q0), color = "yellow", label = "Correct")
    plt.plot(times, network_ham_SV, color = "blue", label = "Network (SV)", linestyle = "dashed")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("KeplerHamiltonianNetwork.pdf", bbox_inches='tight')
    plt.show()

#======================#
# Henon-Heiles Problem #
#======================#

def hehon_heiles_test():
    """Train networks, calculate numerical solutions and plot figures for the Henon-Heiles problem."""
    domain_T = domain_V = [[-2, 2], [-2, 2]]
    HH = Henon_Heiles(domain_T, domain_V)
    d0 = 2
    I = 1000
    d = 4
    K = 30
    h = 0.1
    iterations = 5000
    tau = None
    chunk = int(I/10)

    # Train the neural networks. 
    NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, HH.T, HH.domain_T)
    NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, HH.V, HH.domain_V)

    q0  = p0 = np.array([0.2,0.2])
    timesteps = 5000
    times = np.linspace(0, 20, timesteps)

    # Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
    network_sol_euler = symplectic_euler_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_euler = symplectic_euler_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)
    network_sol_SV = stormer_verlet_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)

    # Plot the exact position and the network position resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times 
    xline = exact_sol_euler[0,:]
    yline = exact_sol_euler[1,:]
    ax.plot3D(xline, yline, zline, "black", label = "Exact (Euler)", linestyle = "dashed")
    xline = exact_sol_SV[0,:]
    yline = exact_sol_SV[1,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_euler[0,:]
    ynet = network_sol_euler[1,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)", linestyle = "dashed") 
    xnet = network_sol_SV[0,:]
    ynet = network_sol_SV[1,:]
    ax.plot3D(xnet, ynet, zline, "blue", label = "Network (SV)") 
    plt.legend()
    fig.suptitle("Position in Time Space", fontweight = "bold")
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("HenonPosition.pdf", bbox_inches='tight')
    plt.show()

    # Plot the exact momentum and the network momentum resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times 
    xline = exact_sol_euler[2,:]
    yline = exact_sol_euler[3,:]
    ax.plot3D(xline, yline, zline, "black", label = "Exact (Euler)", linestyle = "dashed")
    xline = exact_sol_SV[2,:]
    yline = exact_sol_SV[3,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_euler[2,:]
    ynet = network_sol_euler[3,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)", linestyle = "dashed") 
    xnet = network_sol_SV[2,:]
    ynet = network_sol_SV[3,:]
    ax.plot3D(xnet, ynet, zline, "blue", label = "Network (SV)") 
    plt.legend()
    fig.suptitle("Momentum in Time Space", fontweight = "bold")
    ax.set_xlabel("$p_1$")
    ax.set_ylabel("$p_2$")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("HenonMomentum.pdf", bbox_inches='tight')
    plt.show()

    network_ham_euler = NNT.calculate_output(network_sol_euler[d0:,:].reshape(d0,timesteps)) + NNV.calculate_output(network_sol_euler[:d0,:].reshape(d0,timesteps))
    network_ham_SV = NNT.calculate_output(network_sol_SV[d0:,:].reshape(d0,timesteps)) + NNV.calculate_output(network_sol_SV[:d0,:].reshape(d0,timesteps))
    exact_ham_euler = HH.T(exact_sol_euler[d0:,:])+HH.V(exact_sol_euler[:d0,:])
    exact_ham_SV = HH.T(exact_sol_SV[d0:,:])+HH.V(exact_sol_SV[:d0,:])

    # Plot Hamiltonian with exact gradient. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, exact_ham_euler, label = "Exact (Euler)", color = "red", linestyle = "dashed")
    plt.axhline(y = HH.T(p0)+HH.V(q0), color = "yellow", label = "Correct")
    plt.plot(times, exact_ham_SV, color = "blue", label = "Exact (SV)", linestyle = "dashed")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("HenonHeilesHamiltonianExact.pdf", bbox_inches='tight')
    plt.show()

    # Plot Hamiltonian with approximate gradient from network.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_ham_euler, label = "Network (Euler)", color = "red", linestyle = "dashed")
    plt.axhline(y = HH.T(p0)+HH.V(q0), color = "yellow", label = "Correct")
    plt.plot(times, network_ham_SV, color = "blue", label = "Network (SV)", linestyle = "dashed")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("HenonHeilesHamiltonianNetwork.pdf", bbox_inches='tight')
    plt.show()
    


#=====================================#
# Symplectic Euler vs. Størmer-Verlet #
#=====================================#


def SE_vs_SV():
    d0 = 2
    HH = Henon_Heiles(None, None)
    q0 = p0 = np.array([0.2,0.2])
    timesteps = 100
    times = np.linspace(0, 10, timesteps)

    exact_sol_euler = symplectic_euler_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)

    exact_ham_stormer = HH.T(exact_sol_SV[d0:,:])+HH.V(exact_sol_SV[:d0,:])
    exact_ham_euler = HH.T(exact_sol_euler[d0:,:])+HH.V(exact_sol_euler[:d0,:])

    # Plot Hamiltonian.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    fig.suptitle("Hamiltonian in Henon-Heiles", fontweight = "bold")
    plt.plot(times, exact_ham_stormer, color = "blue", label = "størmer-verlet", linewidth = 0.5)
    plt.plot(times, exact_ham_euler, color = "green", label = "symplectic euler", linewidth = 0.5)
    plt.axhline(y = HH.T(p0)+ HH.V(q0), color = "yellow", label = "analytic", linestyle = "dashed")
    plt.legend()
    plt.show()

    d0 = 1
    NP = Pendulum(None, None)
    q0 = p0 = np.array([0.2])

    exact_sol_euler = symplectic_euler_exact(q0, p0, times, NP.grad_T, NP.grad_V, d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, NP.grad_T, NP.grad_V, d0)

    exact_ham_stormer = NP.T(exact_sol_SV[d0:,:])+NP.V(exact_sol_SV[:d0,:])
    exact_ham_euler = NP.T(exact_sol_euler[d0:,:])+NP.V(exact_sol_euler[:d0,:])

    # Plot Hamiltonian.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    fig.suptitle("Hamiltonian in Nonlinear Pendulum", fontweight = "bold")
    plt.plot(times, exact_ham_stormer.reshape(timesteps), color = "blue", label = "SV", linewidth = 0.5)
    plt.plot(times, exact_ham_euler.reshape(timesteps), color = "red", label = "Euler", linewidth = 0.5)
    plt.axhline(y = NP.T(p0)+ NP.V(q0), color = "yellow", label = "correct", linestyle = "dashed")
    plt.legend()
    plt.show()
