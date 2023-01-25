import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


#from sklearn.model_selection import train_test_split
#log function to log the results for every time the model is run.
def log(error,MSE,opt,n_epochs,n_batch,n_layers,n_neurons):
    with open('logs.txt','a')as file:
        file.write(f"{datetime.datetime.now()} - Optimizer: {opt} - Number of layers: {n_layers} - Number of neuron: {n_neurons} - Epoch number: {n_epochs:} - batch_size:{n_batch:} - MSE: {MSE:.4f}\n")


# Data set generation
x=np.arange(0.01,4.01,0.01)
y=(1/12)*x**4-(1/3)*x**3-(7/12)*x**2+(5/6)*x

# Adding noise.
noisey_y=y+np.random.normal(0,1,len(y))

#spliting the data for training and testing

sample_length =round(0.6*len(x))
x_learning, x_testing=x[:sample_length],x[sample_length:]
y_learning, y_testing=noisey_y[:sample_length],noisey_y[sample_length:]

#setting up the model. 
opt='adam'
los='mse'
n_n=4
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(n_n,input_shape=(1,), activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=opt, loss=los)

#training the model. 
n_epochs=450
n_batch=35
model.fit(x_learning,y_learning,epochs=n_epochs,batch_size=n_batch)

#estimation
y_result=model.predict(x_testing)
print(y_result)

#checking for the accuracy of the model
error=y_result-y_testing
MSE=np.mean(error**2)
#print(error)
print(MSE)

log(error, MSE, opt, n_epochs, n_batch,2,n_n)


#plotting 
# plt.plot(x,y)
# plt.plot(x,noisey_y)
# plt.show()

plt.plot(x_testing,y_testing)
plt.plot(x_testing,y_result)
plt.show()