import copy, numpy as np, MachineTest as mt, csv
np.random.seed(0)

machismo = mt.machine(2,2)

nnInput = np.array([[.05,.10]])

machismo.input_synapse.data =  np.array([[.15,.25],[.20,.30]])
machismo.output_synapse.data = np.array([[.40,.50],[.45,.55]])

nnTargt = np.array([[.01,.99]])

machismo.input_bias = .35
machismo.hidden_bias = .6

machismo.train(nnTargt,nnInput)
print ("Input:")
print(str(nnInput))
print ("Hidden:")
print (str(machismo.hidden_net))
print (str(machismo.hidden_layer))
print ("Output:")
print (str(machismo.output_net))
print (str(machismo.output_layer))
