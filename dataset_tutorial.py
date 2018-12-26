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

def generator():
    for el in sequence:
        yield el

sequence_dataset = tf.data.Dataset().from_generator(generator,
                                                    output_types=tf.int64,
                                                    output_shapes=tf.TensorShape([None,1]))
iter = sequence_dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    try:
        while True:
            print(sess.run(el))
    except tf.errors.OutOfRangeError:
        pass
 


#From a csv file
CSV_PATH = './olympics_2016.csv'
csv_dataset = tf.data.TextLineDataset(CSV_PATH)
iter = csv_dataset.make_initializable_iterator() 
next = iter.get_next()                      
with tf.Session() as sess:
    sess.run(iter.initializer)
   try:
       while True:
           print(sess.run(next))
   except tf.errors.OutOfRangeError:
       pass






