import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load
import utils
from tensorflow.examples.tutorials.mnist import input_data

class Networks():
    
    def __init__(self, num_hidden_node, num_class, batch_size, epoch, learning_rate):
        
        '''
        Assign values to model hyperparameters
        '''
        
        self.num_hidden_node = num_hidden_node
        self.num_class = num_class
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.graph = tf.Graph()
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0),trainable=False)
        
    def import_mnist(self):
        mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
        self.test_data, self.test_labels = mnist.test.next_batch(3000)
        self.train = mnist.train
        self.feature_length = self.test_data.shape[1]
        
          
    def build_graph(self):
        '''
        import data
        '''
        
        self.inputs = tf.placeholder(dtype=tf.float32,name='inputs', shape=[None,self.feature_length])
        self.labels = tf.placeholder(dtype=tf.int32,name='labels', shape=[None,self.num_class])
               
        '''
        hidden layer
        hidden layer weigths shape: input_length * num_hidden_node
        hidden layer bias shape: 1 * num_hidden_node
        '''
        self.hidden_layer_weights = tf.get_variable(name='hidden_layer_weights',
                                               shape=[self.inputs.shape[1],self.num_hidden_node],
                                               initializer=tf.truncated_normal_initializer(
                                                   mean=0.0,
                                                   stddev=0.01))
        self.hidden_layer_bias = tf.get_variable(name='hidden_layer_bias',
                                            shape=[1,self.num_hidden_node],
                                            initializer=tf.constant_initializer(0))
                                           
       
        hidden_layer_output = tf.matmul(self.inputs,self.hidden_layer_weights) + self.hidden_layer_bias
        hidden_layer_output = tf.nn.relu(hidden_layer_output)
        
        '''
        output layer
        output layer weights shape: hidden_layer_output.shape[1] * num_class
        output layer bias shape: 1 * num_class
      
        '''
        self.output_layer_weights = tf.get_variable(name='output_layer_weights',
                                               shape=[hidden_layer_output.shape[1], self.num_class],
                                               initializer=tf.truncated_normal_initializer(
                                                   mean=0.0,
                                                   stddev=0.01))
        self.output_layer_bias = tf.get_variable(name='output_layer_bias',
                                            shape=[1, self.num_class],
                                            initializer=tf.constant_initializer(0))
        
        output_layer_output = tf.matmul(hidden_layer_output, self.output_layer_weights) \
                              + self.output_layer_bias

        correct_preds = tf.equal(tf.argmax(output_layer_output, 1), tf.argmax(self.labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) 
        '''
        loss
        '''
        self.loss = tf.losses.softmax_cross_entropy(self.labels,output_layer_output)

        '''optimizer'''
        
        #self.optimizer = tf.train.GradientDescentOptimizer(
            #learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
            
    def train_networks(self):
        #Create a session and start to train and optimize weights and bias

        with tf.Session() as sess:
   
            sess.run(tf.global_variables_initializer())
            
            for i in range(self.epoch):
                total_loss = 0
                step = 0
                accurate = 0
                for j in range (int(self.train.num_examples/self.batch_size)):
                    batch, batch_labels = self.train.next_batch(self.batch_size)
                    one_accurate,oneloss,_= sess.run([self.accuracy, self.loss, self.optimizer],
                                                         feed_dict={self.inputs:batch, self.labels:batch_labels})
                    step += 1
                    total_loss += oneloss
                    

                    if(step % 100 == 0):
                       print('Average loss at step {}, epoch {} : {:5.3f} / {:0.3f}'.
                              format(step, i, oneloss, one_accurate))

            test_accuracy = sess.run(self.accuracy,feed_dict={self.inputs:self.test_data,self.labels:self.test_labels})
            print('test accuracy:{:0.3f}'.format(test_accuracy))
            
                          

if __name__=='__main__':
    num_hidden_node = 100
    num_class = 10
    batch_size = 128
    epoch = 10
    learning_rate = 0.1
    nt = Networks(num_hidden_node,num_class, batch_size,epoch, learning_rate)
    nt.import_mnist()
    nt.build_graph()
    nt.train_networks()
    
    
