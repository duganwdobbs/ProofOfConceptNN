import numpy as np, copy, math

#Synapse class.
class synapse:
    def __init__(self, i, j):
        self.data = 2*np.random.random((i,j))-1
    def doot(self,i):
        return np.dot(i,self.data)
    def update(self,i,learn_rate = .5):
        self.data = self.data - i * learn_rate;
    def getSyn(self,i=-1,j=-1):
        if(i != -1 and j != -1):
            return self.data[i][j]
        else:
            return self.data

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sig_der(x):
    x=np.atleast_2d(x)
    output = np.zeros(x.shape)
    for i in range (x.shape[0]) :
        for j in range (x.shape[1]) :
            output[i][j] = x[i][j] * (1 - x[i][j])
    return output
'''
def dot_product(mat1,mat2):
    #mat1 dot mat2
    output = np.zeros(mat1.shape[0],mat2.shape[1])
    for row in range(mat1.shape[0])
        for col in range(mat2.shape[1])

        '''
class machine:
    def __init__(self,i,j):
        self.input_size  = i
        self.output_size = j
        self.hidden_size = self.input_size
        self.input_synapse   = synapse(self.input_size, self.hidden_size)
        self.output_synapse  = synapse(self.hidden_size,self.output_size)
        self.input_bias = 0
        self.hidden_bias = 0
        self.output_layer = 0

    def test(self,input_layer):
        self.hidden_net = np.dot(      input_layer,self.input_synapse.data) + self.input_bias
        self.hidden_layer = np.atleast_2d(sigmoid(self.hidden_net))
        self.output_net = np.dot(self.hidden_layer,self.output_synapse.data) + self.hidden_bias
        self.output_layer = np.atleast_2d(sigmoid(self.output_net))
        return self.output_layer.flatten()

    def train(self,target,input_layer,learn_rate = .3):
        self.test(input_layer)

        output_deltas = np.dot(self.hidden_layer.T,sig_der(self.output_layer))
        output_error  = self.output_layer[0] - target
        output_deltas = output_error * output_deltas

        #This code works, and I don't know why.
        #It should be input_error = np.dot(output_error * sig_der(output_layer), self.output_synapse.data.T)
        input_deltas = np.dot(input_layer.T,sig_der(self.hidden_layer))  # = [I][H]
        input_error  = np.dot(output_error * sig_der(self.output_layer), self.output_synapse.data.T)  # = [1][H]
        input_deltas = input_error * input_deltas                        # = [I][H] . [1][H] = [I][H]

        print("Output Updates: ")
        print(str(output_deltas))
        print("Input Updates: ")
        print(str(input_deltas))
        self.input_synapse.update(input_deltas)
        self.output_synapse.update(output_deltas)
        return self.output_layer
    def get_error(self):
        return self.output_error
