"""Testing the convergence rate when scaling input and output"""
from network_algo import *
from test_model import *


def compute_avg_scaling(runs, I,d,d0, K, h, iterations, tau, chunk, method, function, domain, scaling, alpha, beta, hypothesis = 1, plot = False): 
    """Computes average J_list after n runs"""
    J_arr = np.zeros((runs, iterations))
    it = None
    for i in range(runs):
        NN, a1, b1, a2, b2, J_list, it = algorithm_scaling(I, d, d0, K, h, iterations, tau, chunk, method, function, domain, scaling, alpha, beta, hypothesis, plot)
        J_arr[i,:] = J_list
       
    return np.average(J_arr, axis = 0), it 


I = 500 # Amount of points ran through the network at once. 
K = 23 # Amount of hidden layers in the network.
d = 2 # Dimension of the hidden layers in the network. 
h = 0.1 # Scaling of the activation function application in algorithm.  
iterations = 10000 #Number of iterations in the Algorithm.
tau = 0.01 #For the Vanilla Gradient method.

# For scaling.
scaling = True
alpha = 0.2
beta = 0.8 

d0 = 1 # Dimension of the input layer. 
domain = [-2,2]
chunk = int(I/10)
def test_function1(x):
    return 1 - np.cos(x)

runs = 5
method = "vanilla"
J_list1, it = compute_avg_scaling(runs, I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)
J_list2, it = compute_avg_scaling(runs, I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, hypothesis = 2, plot = False)

method = "adams"
J_list3, it = compute_avg_scaling(runs, I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)
J_list4, it = compute_avg_scaling(runs, I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, hypothesis = 2, plot = False)
scaling = False
J_list5, it = compute_avg_scaling(runs, I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)

# Plotting convergence.
plt.plot(it, J_list1, label = "Vanilla and scaling. Hypothesis = 1", linewidth = 0.5)
plt.plot(it, J_list2, label = "Vanilla and scaling. Hypothesis = 2", linewidth = 0.5)
plt.plot(it, J_list3, label = "ADAM and scaling. Hypothesis = 1", linewidth = 0.5)
plt.plot(it, J_list4, label = "ADAM and scaling. Hypothesis = 2", linewidth = 0.5)
plt.plot(it, J_list5, label = "ADAM without scaling. Hypothesis = 1", linewidth = 0.5)


plt.yscale("log")
plt.ylabel(r"$\overline{\mathrm{J}}$")
plt.xlabel("Iteration")
plt.legend()
plt.show()

