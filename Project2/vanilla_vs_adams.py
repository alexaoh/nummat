"""Testing convergence rates when using Vanilla gradient descent vs ADAM method."""
from test_model import *
from network_algo import *


def algorithm_input(inp, method, I, d, d0, K, h, iterations, tau, chunk, function, domain):
    """Trains a network based on prespecified input and returns the values of the Objective functions"""
  
    sol = get_solution(function,inp,d,I,d0)
    Z_0 = inp[:,0:chunk]
    c_0 = sol[0:chunk]
    NN = Network(K,d,chunk,h,Z_0,c_0)
    
    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):
        
        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient, method, tau, j)

        if counter < I/chunk - 1:
            NN.Z_list[0,:,:] = inp[:,chunk*(counter+1):chunk*(counter+2)]
            NN.c = sol[chunk*(counter + 1):chunk*(counter + 2)]
        else:
            counter = 0

        J_list[j-1] = NN.J()
        it[j-1] = j

    return J_list, it


def compute_avg(runs, I, d, d0, K, h, iterations, tau, chunk, function, domain): 
    J_adam_arr = np.zeros((runs, iterations))
    J_vanilla_arr = np.zeros((runs, iterations))
    it = None
    for i in range(runs):
        inp = generate_input(f,domain,d0,I,d)
        J_vanilla, it = algorithm_input(inp, "vanilla", I, d, d0, K, h, iterations, tau, chunk, f, domain)
        J_adam, it = algorithm_input(inp, "adams", I, d, d0, K, h, iterations, tau, chunk, f, domain)  

        J_adam_arr[i,:] = J_adam
        J_vanilla_arr[i,:] = J_vanilla

    return np.average(J_adam_arr, axis = 0), np.average(J_vanilla_arr, axis = 0), it 

        

I = 1000 # Amount of points ran through the network at once. 
K = 20 # Amount of hidden layers in the network.
d = 2 # Dimension of the hidden layers in the network. 
h = 0.1 # Scaling of the activation function application in algorithm.  
iterations = 5000 #Number of iterations in the Algorithm 

d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
chunk = int(I/10)

def f(x):
    return 0.5*(x[0]**2 + x[1]**2)


tau = 0.005

runs = 15
J_adam_avg, J_vanilla_avg, it = compute_avg(runs, I, d, d0, K, h, iterations, tau, chunk, f, domain)

plt.plot(it, J_vanilla_avg, label = "Vanilla")
plt.plot(it, J_adam_avg, label = "ADAM")
plt.xlabel("Iteraion")
plt.ylabel(r"$\overline{\mathrm{J}}$")
plt.yscale("log")
plt.legend()
plt.show()
