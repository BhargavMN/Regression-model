import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#from sklearn.model_selection import train_test_split



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
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')

#training the model. 
model.fit(x_learning,y_learning,epochs=100,batch_size=32)

#estimation
y_result=model.predict(x_testing)
#checking for the accuracy of the model
error=y_result-y_testing
MSE=np.square(np.mean(error))


#plotting 
#plt.plot(x,y)
#plt.plot(x,noisey_y)
#plt.show()

# plt.plot(x_testing,y_testing)
# plt.plot(x_testing,y_result)
# plt.show()