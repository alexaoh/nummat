"""Testing framework for running systematic tests of the neural network.

Ran on the four suggested test functions.

This is in order to find the optimal values of the parameters in the network, 
by method of experimentation (systematic tests).

Parameters tested systematically:
- K
- tau
- d 
- h

Used a fixed amount of iterations and images in each test (see report for further justification).

The code is terribly written, with a lot of copy-paste. This was done for ease of running code 
on Markov for the time being, and is not written for anyones viewing pleasure. 
Could have been vastly improved and cleaned up, but it worked, which was the main focus of the
testing framework in general. 
"""
from network_algo import *
import pickle

# Defined globally to make it easier in vim.
K_values = [10, 15, 20, 23, 30]
d_values = [2, 3, 4]
h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
tol = 0.005
iterations = 2000
I = 1000

def training_test_function1(filename, sgd = False):
    """Only training the test function 1."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    tau = 1
    def test_function1(x):
        return 0.5*x**2

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function1, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function1, domain) 
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.
                del NN # Safety measure to avoid leak. 

    # Save data to file.
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def testing_test_function1(filename, tol = tol, sgd = False):
    """Training and testing random data with test function 1."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    tau = 1
    def test_function1(x):
        return 0.5*x**2

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function1, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function1, domain) 
                test_input = generate_input(test_function1,domain,d0,I,d)
                testing(NN, test_input, test_function1, domain, d0, d, I)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])

                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J_last, ratio]
                               
                del NN # Safety measure to avoid leak. 

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def training_test_function2(filename, sgd = False):
    """Only training the test function 2."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    chunk = int(I/1)
    tau = 1
    def test_function2(x):
        return 1 - np.cos(x)

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function2, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function2, domain) 
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.            
                del NN # Safety measure to avoid leak. 

    # Save data to file.
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def testing_test_function2(filename, tol = tol, sgd = False):
    """Training and testing random data with test function 2."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    tau = 1
    def test_function2(x):
        return 1 - np.cos(x)

    config = {}
    i = 1
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function2, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function2, domain) 
                test_input = generate_input(test_function2,domain,d0,I,d)
                testing(NN, test_input, test_function2, domain, d0, d, I)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])

                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J_last, ratio]
                del NN # Safety measure to avoid leak. 

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def training_test_function3(filename, sgd = False):
    """Only training the test function 3."""
    d0 = 2
    d = 4
    tau = 1
    domain = [[-2,2],[-2,2]]
    tau = 1

    def test_function3(x):
        return 0.5*(x[0]**2 + x[1]**2)

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function3, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function3, domain) 
                # The below adds the J_list from training. 
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.           
                del NN # Safety measure to avoid leak.

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()
    
def testing_test_function3(filename, tol = tol, sgd = False):
    """Training and testing random data with test function 3."""
    d0 = 2
    tau = 1
    d = 4
    domain = [[-2,2],[-2,2]]
    tau = 1

    def test_function3(x):
        return 0.5*(x[0]**2 + x[1]**2)

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function3, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function3, domain) 
                test_input = generate_input(test_function3,domain,d0,I,d)
                testing(NN, test_input, test_function3, domain, d0, d, I)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])
            
                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J_last, ratio]
                del NN # Safety measure to avoid leak. 

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def training_test_function4(filename, sgd = False):
    """Only training the test function 4."""
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    tau = 1

    def test_function4(x):
        return -1/np.sqrt(x[0]**2 + x[1]**2)

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function4, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function4, domain) 
                # The below adds the J_list from training. 
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.               
                del NN # Safety measure to avoid leak. 

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def testing_test_function4(filename, tol = tol, sgd = False):
    """Training and testing random data with test function 4."""
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    tau = 1

    def test_function4(x):
        return -1/np.sqrt(x[0]**2 + x[1]**2)

    config = {}
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                if sgd:
                    chunk = int(I/10)
                    NN = algorithm_sgd(I,d,d0,K,h,iterations,tau, chunk, test_function4, domain) 
                else:
                    NN = algorithm(I,d,d0,K,h,iterations,tau, test_function4, domain) 
                test_input = generate_input(test_function4,domain,d0,I,d)
                testing(NN, test_input, test_function4, domain, d0, d, I)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])
            
                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J_last, ratio]
                del NN # Safety measure to avoid leak. 

    # Save data to file. 
    filenm = 'data/markov/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()
    