#import numpy as np
import cupy as np
import math
#input_values = np.array([[1,2,5]])
rng = np.random.default_rng()
max_itterations = 400

class layer:
  def __init__(self, weights:np.array, biases:np.array, activation_function:str, stage:str):
    self.weights = weights
    self.biases = biases
    self.output = np.array([[]])
    self.input = np.array([[]])
    self.activation_function = activation_function
    self.stage = stage

  def forward (self, input):
    #print (f"forward, {self.stage}")
    self.input = np.array(input)
    #print (f"input: {self.input}")
    #print (f"input = {self.input}, weights = {self.weights}")
    self.arr = np.dot(self.input, self.weights)
    self.linear_output = np.dot(self.input, self.weights) + self.biases
    #RELU activation function
    if self.activation_function == "RELU":
       self.output = np.maximum(0, self.linear_output)
    elif self.activation_function == "Tanh":
       self.output = np.tanh(self.linear_output)
    else:
       self.output = self.linear_output
    #print (f"output, {self.output}")
    return (self.output)
  def back_propagation (self, learning:float, upstream_gradient:np.array):
    if self.activation_function == "RELU":
      self.deriv = np.where(self.output > 0, 1, 0)
    if self.activation_function == "Tanh":
      self.deriv =  (1 - np.power(self.output, 2))
    else:
      self.deriv = np.ones_like(self.output)
    #print (f"backpropergation, {self.stage}")
    #print (f"input, {self.input.shape}")
   # print (f"self.deriv, {self.deriv.shape}")
   # print (f"self.weights, {self.weights.shape}")
   # print (f"upstream grad, {upstream_gradient.shape}")
    self.delta = upstream_gradient * self.deriv
   # print (f"self.delta, {self.delta.shape}")
    self.weights = self.weights-(learning* np.dot(self.input.T, self.delta))
    self.biases = self.biases-(learning*np.sum(self.delta, axis=0, keepdims=True))
    return np.dot(self.delta, self.weights.T)
class model:
  def __init__ (self, params:dict, n_features:int, learning_rate:float, batch_size:int):
    self.batch_size=batch_size
    self.pipeline = []
    self.learning =learning_rate
    self.params = params
    self.sum_abselute_error = 0
    self.analytics={}
    self.analytics["MAE"]=[]
    self.n_features = n_features
    self.func_table = {
        "input":self.type_input,
        "output":self.type_output,
        "hidden":self.type_hidden,
        "random_b":self.random_b,
        "random_w":self.random_w
    }
    for i in range (len (self.params)):
        self.layer = self.params[i]
        layer_type = self.func_table.get(self.layer["type"])
        layer_type()
        weights_init = self.func_table.get(self.layer["weights_init"])
        weights_init()
        biases_init = self.func_table.get(self.layer["biases_init"])
        biases_init()
        
        self.params[i]["layer_obj"]=layer(weights=self.weights, biases=self.biases, activation_function=params[i]["activation_function"], stage = self.layer["type"])
  def random_w(self):
    rng = np.random.default_rng()
    self.weights = rng.standard_normal(size=(self.m, self.n)) * np.sqrt(0.1 / 4)
  def random_b(self):
    rng = np.random.default_rng()
    if self.layer["type"] == "output":
      self.biases=rng.uniform(-0.1,0.1, size =(1, 1))
    else:
      self.biases=rng.uniform(-0.1,0.1, size =(1, self.layer["hidden_features"]))
  def type_input (self):
    self.m = self.n_features
    self.n = int(self.layer["hidden_features"])
    print (f"m,n input = {self.m},{self.n} {type(self.n)}")
  def type_output(self):
    self.n = 1
    self.m = int(self.layer["hidden_features"])
    print (f"m,n output = {self.m,self.n} {type(self.n)}")
  def type_hidden(self):
    self.n = self.layer["hidden_features"]
    self.m = self.layer["hidden_features"]
    print (f"m,n = hidden {self.m},{self.n} {type(self.n)}")

  def loss_function(self, label:np.array, result:np.array):
    y = np.asarray(label)
    individual_loss_gradient = 2*(result-y)
    self.sum_abselute_error = np.mean(self.sum_abselute_error+((result-y)**2))
    return np.mean(individual_loss_gradient)


  def train(self, y:np.array, x:np.array, epochs:int):
    x = np.asarray(x)
    y = np.asarray(y)
    rows,_=x.shape
    for epoch in range(epochs):
        for p in range(0, rows, self.batch_size):
            Y = y[p:p+(self.batch_size)]
            feed = x[p:p+(self.batch_size)]
            feed_out = [feed]
            for i in range(len(self.params)):
                z = self.params[i]["layer_obj"].forward(feed_out[-1])
                feed_out.append(z)
            output = feed_out[-1]
            loss_grad = self.loss_function(label=Y, result=output)
            loss_grad = np.asarray(loss_grad)
            for i in range(len(self.params) - 1, -1, -1):
                loss_grad = self.params[i]["layer_obj"].back_propagation(upstream_gradient=loss_grad, learning=self.learning)
        
        self.analytics["MAE"].append(float(self.sum_abselute_error/rows))
        self.sum_abselute_error = 0

  def MAE (self):
    return (self.analytics["MAE"])
    
  def predict(self, x):
    feed_out = np.asarray(x)
    for i in range(len(self.params)):
      feed_out = self.params[i]["layer_obj"].forward(feed_out)
    return feed_out




