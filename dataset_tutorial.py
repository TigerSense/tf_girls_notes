import numpy as np
import tensorflow as tf

'''
Importing data
'''

#From numpy
x = np.random.sample([100,2])
x_dataset = tf.data.Dataset.from_tensor_slices(x)

feature, labels = (np.random.sample([100,2]), np.random.sample([100,1]))
f_dataset = tf.data.Dataset.from_tensor_slices((feature, labels))

#From tensor
y = tf.random_uniform([100,2])
y_dataset = tf.data.Dataset.from_tensor_slices(y)

#From placeholder
z = tf.placeholder(dtype=tf.float32, shape=[None,2])
z_dataset = tf.data.Dataset.from_tensor_slices(z)

#From generator
sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])
