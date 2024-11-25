import numpy as np
x = np.array(([2,9],[1,5],[3,6]),dtype=float)
print(x)
y = np.array(([92],[86],[89]),dtype=float)
y=y/100
print(x)
print(y)
def sigmoid(x):
  return (1/(1+ np.exp(-x)))
def derivatives_sigmoid(x):
  return x*(1-x)
epoch = 1000
lr = 0.01
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1,hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
  hinp1 = np.dot(x,wh)
  hinp = hinp1 + bh
  hlayer_act = sigmoid(hinp)
  outinp1 = np.dot(hlayer_act,wout)
  outinp = outinp1 + bout
  output = sigmoid(outinp)
  EO = y - output
outgrad = derivatives_sigmoid(output)
d_output = EO * outgrad
EH = d_output.dot(wout,T)
hiddengrad = derivatives_sigmoid(hlayer_act)
d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output)*lr
bout += np.sum(d_output, axis = 0, keepdims = True)*lr
wh += x.T.dot(d_hiddenlayer)*lr
bh += np.sum(d_hiddenlayer, axis = 0, keepdims = True)*lr
print("Actual Output:\n"+ str(y))
print("Predicted Output:\n"+ str(output))
print("Error" + str(EO))
