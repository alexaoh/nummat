"""Network class and main algorithm for training the network."""
from test_model import *
import random

# Add parameters for plotting. 
mpl.rcParams['figure.titleweight'] = "bold"
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.titleweight'] = "bold"

plt.style.use('seaborn')

class Parameters:
    """Class for the parameters in the Neural Network."""
    def __init__(self,K,d,I):
        self.W_k = np.random.randn(K,d,d)
        self.b_k = np.random.randn(K,d,1)
        self.b_k_I = np.zeros((K,d,I))

        for i in range(K):
            self.b_k_I[i,:,:] = self.b_k[i,:,:]

        self.w = np.random.randn(d)  
        self.my = np.random.rand()
        self.K = K
        
        # For Adam descent.
        self.beta_1 = 0.9; self.beta_2 = 0.999; self.alpha = 0.01; self.epsilon = 10E-8
        self.v = [0]; self.m = [0]
         
    def update_parameters(self,gradient,method,tau,j):
        """Update the parameters via Vanilla Gradient Method or Adam Descent."""
        if(method == "vanilla"):
            self.W_k = self.W_k - tau*gradient[0]
            self.b_k = self.b_k - tau*gradient[1]
            for i in range(self.K):
                self.b_k_I[i,:,:] = self.b_k[i,:,:]
            self.w = self.w - tau*gradient[2]
            self.my = self.my - tau*gradient[3]
            
        elif(method == "adams"):
            g = gradient 
            self.m.append( self.beta_1*self.m[j-1] + (1-self.beta_1)*g)
            self.v.append(self.beta_2*self.v[j-1]+(1-self.beta_2)*(g*g))
            m_hat = self.m[j]/(1-self.beta_1**j)
            v_hat = self.v[j]/(1-self.beta_2**j)
            subtract = self.alpha*m_hat/(self.root(self.v[j])+self.epsilon)
            
            self.W_k = self.W_k - subtract[0]
            self.b_k = self.b_k - subtract[1]
            for i in range(self.K):
                self.b_k_I[i,:,:] = self.b_k[i,:,:]
            self.w = self.w - subtract[2]
            self.my = self.my - subtract[3]      
                      
        else:
            print("No method found") 
    
    def root(self, v_j):
        """Calculate the root of v_j componentwise for Adam descent."""
        for i in range(4):
            v_j[i] = np.sqrt(v_j[i])
        return v_j

class Network:
    """Class for the Neural Network (ResNet)."""
    def __init__(self,K,d,I,h,Z_0,c, hypothesis = 1):
        self.theta = Parameters(K,d,I)
        self.Z_list = np.zeros((K+1,d,I))
        self.Z_list[0,:,:] = Z_0
        self.c = c
        self.h = h
        self.K = K
        self.I = I
        self.d = d
        self.Y = None
        self.hypothesis = hypothesis

        self.J_last = None # For use in the testing (instead of returning J from algorithm, save in NN)
    
    def J(self): 
        """Objective function."""
        return 0.5 * la.norm(self.Y - self.c)**2
    
    def sigma(self, x):
        """Sigmoid activation function."""
        return np.tanh(x)

    def sigma_der(self, x):
        """Derivative of the sigmoid activation function."""
        val = 1-np.tanh(x)**2
        return val

    def eta(self, x):
        """Hypothesis function. Appears in the final layer of the neural network."""
        if self.hypothesis == 1:
            val = x
        elif self.hypothesis == 2:
            val = 0.5*(1+ np.tanh(x/2))
        return val

    def eta_der(self, x):
        """The derivative of the hypothesis function."""
        if self.hypothesis == 1:
            val = np.ones(x.shape)
        elif self.hypothesis == 2:
            val = 0.25*self.sigma_der(x/2) 
        return val

    def forward_function(self): 
        """Calculate Y."""
        for i in range(self.K):
            self.Z_list[i+1,:,:] = self.Z_list[i,:,:] + \
                self.h*self.sigma(self.theta.W_k[i,:,:] @ \
                    self.Z_list[i,:,:] + self.theta.b_k_I[i,:,:])
            
        ZT_K = np.transpose(self.Z_list[-1,:,:]) 
        one_vec = np.ones(self.I)
        Y = self.eta(ZT_K @ self.theta.w + self.theta.my*one_vec)
        self.Y = Y

    def back_propagation(self):
        """Calculate and return the gradient of the objective function."""
        ZT_K = np.transpose(self.Z_list[-1,:,:])
        one_vec = np.ones(self.I)
        J_der_my =  self.eta_der(ZT_K @ self.theta.w + self.theta.my*one_vec).T @ (self.Y-self.c)
        J_der_w = self.Z_list[-1,:,:] @ ((self.Y-self.c) * \
                            self.eta_der(ZT_K @ self.theta.w + self.theta.my*one_vec))
        
        P_K = np.outer(self.theta.w,(self.Y-self.c)*self.eta_der(ZT_K @ \
                                            self.theta.w + self.theta.my*one_vec)) 
        
        
        P_list = np.zeros((self.K,self.d,self.I)) 
        P_list[-1,:,:] = P_K 
        for i in range(self.K-1,0,-1):
            P_list[i-1,:,:] = P_list[i,:,:] + self.h*np.transpose(self.theta.W_k[i-1,:,:]) @ \
            (self.sigma_der(self.theta.W_k[i-1,:,:] @ self.Z_list[i-1,:,:] + \
                            self.theta.b_k_I[i-1,:,:]) * P_list[i,:,:])

        J_der_Wk = np.zeros((self.K,self.d,self.d))
        J_der_b = np.zeros((self.K,self.d,1))
        one_vec = np.ones((self.I,1)) 
        
        for i in range(self.K):
            val = P_list[i,:,:] * self.sigma_der(self.theta.W_k[i,:,:] @ \
                                                 self.Z_list[i,:,:] + self.theta.b_k_I[i,:,:])
            J_der_Wk[i,:,:] = self.h*(val @ np.transpose(self.Z_list[i,:,:]))
            J_der_b[i,:,:] = self.h*(val @ one_vec)
        
        gradient = np.array((J_der_Wk,J_der_b,J_der_w,J_der_my))
        
        return gradient
    
    def embed_input_and_sol(self, inp, sol):
        """Embed the input into d-dimensional space. This function is used for testing."""
        self.I = inp.shape[1]
        d0 = inp.shape[0]
        if d0 < self.d:
            while d0 < d:
                zero_row = np.zeros(self.I)
                inp = inp.vstack((inp,zero_row))
                d0 += 1
       

        self.Z_list = np.zeros((self.K+1,self.d,self.I))
        self.Z_list[0,:,:] = inp
        self.c = sol


        self.theta.b_k_I = np.zeros((self.K,self.d,self.I))
        for i in range(self.K):
            self.theta.b_k_I[i,:,:] = self.theta.b_k[i,:,:]
    
    def embed_input(self, inp):
        """ Embed the input in d-dimenstional space when no solution is provided.""" 
        if inp.ndim == 1: # The input is a point.
            self.I = 1
            d0 = inp.shape[0]
            inp = inp.reshape(d0, 1)
        else:
            d0 = inp.shape[0]
            self.I = inp.shape[1]

        self.Z_list = np.zeros((self.K+1,self.d,self.I))
        self.Z_list[0,:d0,:] = inp
        
        # Changing b_K_I in case self.I has changed.
        self.theta.b_k_I = np.zeros((self.K,self.d,self.I))
        for i in range(self.K):
            self.theta.b_k_I[i,:,:] = self.theta.b_k[i,:,:]

    def calculate_output(self, inp):
        """Calculate and return the network's output from a given input."""
        self.embed_input(inp)
        self.forward_function()
        return self.Y

    def Hamiltonian_gradient(self):
        """Calculate the gradient of F, according to the theoretical derivation.
        
        In general it is used for the entire Hamiltonian F, but since we are 
        working with separable Hamiltonians, it can be used on both T and V.
        """
        one_vec = np.ones((self.I,1)) 

    
        w = self.theta.w.reshape((self.d,1)) # Need different dimensions for w. 
        A = w @ self.eta_der(w.T@self.Z_list[self.K,:,:] + self.theta.my*one_vec.T) 
                                                    

        for k in range(self.K, 0, -1):
            A += self.theta.W_k[k-1,:,:].T @ (self.h*self.sigma_der(\
                    self.theta.W_k[k-1,:,:]@self.Z_list[k-1,:,:] + self.theta.b_k_I[k-1,:,:])*A)
        
        return A

def algorithm(I, d, d0, K, h, iterations, tau, function, domain, plot = False, savename = ""):
    """Main training algorithm."""
      
    inp = generate_input(function,domain,d0,I,d)
    output = get_solution(function,inp,d,I,d0)

    Z_0 = inp
    c_0 = output
    NN = Network(K,d,I,h,Z_0,c_0)
    
    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    for j in range(1,iterations+1):
        
        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient,"adams",tau,j)

        J_list[j-1] = NN.J()
        it[j-1] = j
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(it,J_list)
        plt.yscale("log")
        fig.suptitle("Objective Function J as a Function of Iterations.", fontweight = "bold")
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 6)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename+".pdf", bbox_inches='tight')
        plt.show()
    
    
    NN.J_last = J_list[-1] # Save last value of J_list in NN, to check which converges best in tests. 
    
    return NN

def algorithm_sgd(I,d, d0, K, h, iterations, tau, chunk, function, domain, plot = False, savename = ""):
    """Main training algorithm with Stochastic Gradient Descent."""

    inp = generate_input(function,domain,d0,I,d)
    output = get_solution(function,inp,d,I,d0)

    index_list = [i for i in range(I)]

    Z_0, c_0, index_list = get_random_sample(inp,output,index_list,chunk,d)
    NN = Network(K,d,chunk,h,Z_0,c_0)

    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):

        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient,"adams",tau,j)

        # For plotting.
        J_list[j-1] = NN.J()
        it[j-1] = j

        if counter < I/chunk - 1:
            Z, c, index_list = get_random_sample(inp,output,index_list,chunk,d)
            NN.Z_list[0,:,:] = Z
            NN.c = c
            counter += 1
        else:
            # All data has been sifted through.
            counter = 0
            index_list = [i for i in range(I)]


    if plot:
        fig, ax = plt.subplots()
        ax.plot(it,J_list)
        plt.yscale("log")
        fig.suptitle("Objective Function J as a Function of Iterations.", fontweight = "bold")
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 6)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename + ".pdf", bbox_inches='tight')
        plt.yscale("log")
        plt.show()
    return NN 

def algorithm_scaling(I,d, d0, K,h,iterations, tau, chunk, method, function,domain,scaling, alpha, beta, hypothesis = 1, plot = False, savename = ""):
    """Main training algorithm with SGD and option to scale."""

    inp = generate_input(function,domain,d0,I,d)
    output = get_solution(function,inp,d,I,d0)

    a1 = b1 = a2 = b2 = None
    if scaling:
        inp, a1, b1 = scale_data(alpha,beta,inp)
        output, a2, b2 = scale_data(alpha,beta,output)

    index_list = [i for i in range(I)]

    Z_0, c_0, index_list = get_random_sample(inp,output,index_list,chunk,d)
    NN = Network(K,d,chunk,h,Z_0,c_0)

    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):


        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient,method,tau,j)

        # For plotting.
        J_list[j-1] = NN.J()
        it[j-1] = j

        if counter < I/chunk - 1:
            Z, c, index_list = get_random_sample(inp,output,index_list,chunk,d)
            NN.Z_list[0,:,:] = Z
            NN.c = c
            counter += 1
        else:
            # All data has been sifted through.
            counter = 0
            index_list = [i for i in range(I)]


    if plot:
        fig, ax = plt.subplots()
        ax.plot(it,J_list)
        fig.suptitle("Objective Function J as a Function of Iterations.", fontweight = "bold")
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 4)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename + ".pdf", bbox_inches='tight')
        plt.show()
        
    NN.J_last = J_list[-1] # Save last value of J_list in NN, to check which converges best in tests. 
    return NN, a1, b1, a2, b2, J_list, it
