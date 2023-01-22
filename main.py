import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data set generation
x=np.arange(0.01,4.01,0.01)
y=(1/12)*x**4-(1/3)*x**3-(7/12)*x**2+(5/6)*x

# Adding noise.
noisey_y=y+np.random.normal(0,1,len(y))

#spliting the data for training and testing

X_training, X_testing, y_training, y_testing = train_test_split(
    x, y, test_size=0.4
)

#plotting 
plt.plot(x,y)
plt.plot(x,noisey_y)
plt.show()
