"""Utilities used to run training and tests on the networks."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy.linalg as la
from mpl_toolkits import mplot3d
import random

# Add parameters for plotting. 
mpl.rcParams['figure.titleweight'] = "bold"
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.titleweight'] = "bold"

def testing(Network, test_input, function, domain, d0, d, I):
    """Testing the Neural Network with new random data. 
    
    The parameters found from the training of the Neural Network are employed.
    """
    test_output = get_solution(function, test_input, d, I, d0)

    Network.embed_input_and_sol(test_input, test_output)
    Network.forward_function()
    output = Network.Y
    return output

def generate_input(function,domain,d0,I,d):
    """Generate and embed input in d-dimensional space.
    
    Done by drawing from a uniform distribution on the domain.
    """

    result = np.zeros((d,I))
    if d0 == 1:
        for i in range(I):
            num = np.random.uniform(domain[0],domain[1])
            result[:d0,i] = num
    if d0 == 2:
        for i in range(I):
            num = np.random.uniform(domain[0][0],domain[0][1])
            result[0,i] = num
        for i in range(I):
            num = np.random.uniform(domain[1][0],domain[1][1])
            result[1,i] = num
    return result

def get_solution(function,inp_values,d,I,d0):
    """Generate points from the test function on the given domain."""
    result = np.zeros(I)

    for i in range(I):
        result[i] = function(inp_values[:d0,i])

    return result
    
def plot_graph_and_output(output,inp,function,domain,d0,d, savename = ""):
    """Plot results from testing the network together with the analytical graph."""
    if d0 == 1:
        # Plot the output from the network.
        x = inp[0,:]
        
        fig, ax = plt.subplots()
        ax.scatter(x,output, color="orange", label="Network")
        fig.suptitle("Analytical Solution Compared to Output From Network", fontweight = "bold")
        ax.set_xlabel("Domain [y]")
        ax.set_ylabel("F(y)")

        # Plot analytical solution.
        x = np.linspace(domain[0],domain[1])
        ax.plot(x,function(x), color="blue", label="Function")
        ax.legend()
        if savename != "":
            plt.savefig(savename+".pdf", bbox_inches='tight')
        plt.show()

    elif d0 == 2:
        # Plot output from network. 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        zdata = output
        xdata = inp[0,:]
        ydata = inp[1,:]
        
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')
        
        # Plot analytical solution.
        x = np.linspace(domain[0][0],domain[0][1], 30)
        y = np.linspace(domain[1][0], domain[1][1], 30)
        
        X, Y = np.meshgrid(x, y)
        Z = function([X, Y])
        fig.suptitle("Analytical Solution Compared to Output From Network", fontweight = "bold")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='Greys', edgecolor='none', alpha = 0.5)
        if savename != "":
            plt.savefig(savename+".pdf", bbox_inches='tight')
        plt.show()


# Utilites for scaling follow. 

def scale_data(alpha, beta, inp):
    """Scale the input relative to alpha and beta."""
    a = np.min(inp)
    b = np.max(inp)
    dim = inp.shape
        
    
    def max_min(dim,inp,a,b,alpha,beta):
        """Max-min transformation."""
        if len(dim) == 1:
            inp = 1/(b-a) * ((b*np.ones(dim[0]) - inp)*alpha \
                                    + (inp - a*np.ones(dim[0]))*beta)
        else:
            for i in range(dim[1]):
                inp[:,i] = 1/(b-a) * ((b*np.ones(dim[0]) - inp[:,i])*alpha \
                                    + (inp[:,i] - a*np.ones(dim[0]))*beta)
        return inp

    return max_min(dim,inp,a,b,alpha,beta), a, b


def scale_up(a, b, alpha, beta, data):
    """The inverse of the min-max transformation."""
    dim = data.shape
    if len(dim) == 1:
        data = 1/(beta-alpha) * ((b-a)*np.ones(dim[0])*data - np.ones(dim[0])*(b*alpha - a*beta))
    else:
        for i in range(dim[1]):
            data[i] = 1/(beta-alpha) * ((b-a)*np.ones(dim[0])*data[i] - np.ones(dim[0])*(b*alpha - a*beta))
    
    return data

def get_random_sample(inp, sol, index_list, chunk, d):
    """Get random sample from input of size chunk and update sol nd index_list."""
    sample = np.zeros((d,chunk))
    sample_sol = np.zeros(chunk)
    random_indices = random.sample(index_list,chunk)

    for i in range(chunk):
        rand_index = random_indices[i]
        index_list.remove(rand_index)
        sample[:,i] = inp[:,rand_index]
        sample_sol[i] = sol[rand_index]

    return sample, sample_sol, index_list
    