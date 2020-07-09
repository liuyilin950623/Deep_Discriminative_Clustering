from keras.datasets import mnist
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(10)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))[:2000]
y = np.concatenate((y_train, y_test))[:2000]
x = np.divide(x, 255.)

print("MNIST dataset loaded, scaled but not flattened, available as x and y.")