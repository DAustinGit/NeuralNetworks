import pandas as pd
from sklearn.model_selection import train_test_split
import cupy as np
import neuralNetwork as nn
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
print (df.head())

Y = df['logS']   # Y is feature used for training
X = df.drop('logS', axis=1)   #X is the feature we are predicting drop logS and axis = 1 --> work in coluns
print (Y.head())
print (X.head())



X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size= 0.20, random_state = 100) #splits the data for train and test sets testing set is 20% of data
#print (X_train)
x_train = X_train.to_numpy()
x_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()
y_max = np.max(Y_train)
x_mean = np.mean(x_train, axis=0, keepdims=True)

y_max_test = np.max(Y_test)
x_mean_test = np.mean(x_test, axis=0, keepdims=True)
x_std_test = np.std(x_test, axis=0, keepdims=True)

x_std = np.std(x_train, axis=0, keepdims=True)
x_train_scaled = (x_train-x_mean)/x_std
Y_train_scaled = Y_train/y_max
x_test_scaled = (x_test-x_mean_test)/x_std_test
Y_test_scaled = Y_test/y_max_test


print (x_train_scaled)


params = {
   0:{"type":"input" , "biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   1:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   2:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   3:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   4:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   5:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   6:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"RELU"},
   7:{"type":"hidden" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"Relu"},
   8:{"type":"output" ,"biases_init":"random_b", "weights_init":"random_w", "hidden_features":200, "activation_function":"LINEAR"}
}


pipeline = nn.model(params=params, n_features=4, learning_rate=1e-9, batch_size=1)
print("training model...")
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
pipeline.train(x=x_train_scaled, y=Y_train, epochs=2000, )


rows_test,_=x_test_scaled.shape
rows_train,_=x_train_scaled.shape



for i in range (rows_test):
 print (f" predicted test y: {pipeline.predict (x_test_scaled[i:i+1])} label: {Y_test[i]}")
rows,_=x_train_scaled.shape

for i in range (rows_train):
  print (f" predicted train y: {pipeline.predict (x_train_scaled[i:i+1])} label: {Y_train[i]}")




MAE = pipeline.MAE()
print (f"mae {MAE}")
plt.plot(MAE, marker='o', color='purple')

plt.title("X-Axis Automatically Maps to List Index")
plt.xlabel("Index (0 to 4)")
plt.ylabel("Value")
plt.grid(True)
plt.show()