# tf_girls_notes

File description:

load_data.py: implments the methods in lecture 5 (data preprocessing) and lecture 6 (label distribution). My implementation doesn't strictly follow tf girls' codes but has some improvements.

Notes of tf_girls

1. How does this one-hot coding np.eye(n_labels)[target_vector] works?
https://stackoverflow.com/questions/45068853/how-does-this-one-hot-vector-conversion-work

2. load mnist dataset (Tuesday afternoon, 12-25-2018, finished mnist_fc_placeholder.py)

    There are two ways to load mnist dataset. One way (deprecated) is to 

       from tensorflow.examples.tutorials.mnist import input_data
 
       mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
       test_samples, test_labels = mnist.test.next_batch(1000)
       
       mnist.test.num_examples return number of elements in test. Same for train.
       
    A very good example can be found here https://github.com/snehalvartak/MNIST/blob/master/FullyConnectedNet.ipynb
 
    Another way is to load mnist from keras
 
       mnist = tf.keras.datasets.mnist
  
       (x_train, y_train),(x_test, y_test) = mnist.load_data()
 
 3. How to visualize intermediate and final results 
 
 4. Tuesday night, 12-25-2018, Tutorial of "How to use Dataset in Tensorflow" (finished the tutorial reading and dataset_tutorial.py)
 
    link: https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428 (In this tutorial, all the examples mix    placeholder with dataset which I think totally lost the point of using dataset. Demos in cs20i are more proper examples. )
    
    cs20i compares placeholder and dataset in https://docs.google.com/document/d/1kMGs68rIHWHifBiqlU3j_2ZkrNj9RquGTe8tJ7eR1sE/edit
    
 5. Wednesday, 12-26-2018, Complete mnist classification using dataset rather than placeholder (mnist_fc_dataset.py)
