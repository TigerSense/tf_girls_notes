# tf_girls_notes

File description:

load_data.py: implments the methods in lecture 5 (data preprocessing) and lecture 6 (label distribution). My implementation doesn't strictly follow tf girls' codes but has some improvements.

Notes of tf_girls

1. How does this one-hot coding np.eye(n_labels)[target_vector] works?
https://stackoverflow.com/questions/45068853/how-does-this-one-hot-vector-conversion-work

2. load mnist dataset

    There are two ways to load mnist dataset. One way (deprecated) is to 

       from tensorflow.examples.tutorials.mnist import input_data
 
       mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
       test_samples, test_labels = mnist.test.next_batch(1000)
       
       mnist.test.num_examples return number of elements in test. Same for train.
 
    Another way is to load mnist from keras
 
       mnist = tf.keras.datasets.mnist
  
       (x_train, y_train),(x_test, y_test) = mnist.load_data()
 
 
