import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load
import utils


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
        
    def import_dataset(self):
        train = load('train_32x32.mat')
        test = load('test_32x32.mat')
        train_data = train['X']
        train_labels = train['y']
        test_data = test['X']
        test_labels = test['y']

        train_data = np.transpose(train_data, [3,0,1,2])
        train_data = utils.rgb2gray(train_data)
        train_data = utils.normalize(train_data,-1,1)
        train_shape = (train_data.shape[0], train_data.shape[1]*train_data.shape[2])
        train_data = np.reshape(train_data, train_shape)
        train_labels = utils.one_hot_coding(train_labels)

        test_data = np.transpose(test_data,[3,0,1,2])
        test_data = utils.rgb2gray(test_data)
        test_data = utils.normalize(test_data,-1,1)
        test_shape = (test_data.shape[0], test_data.shape[1]*test_data.shape[2])
        test_data = np.reshape(test_data, test_shape)
        test_labels = utils.one_hot_coding(test_labels)
        self.im_size = train_data.shape[1]

        #Create datasets from the above tensors
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
      
          
    def build_graph(self):
        '''
        import data
        '''
        self.iterator = self.train_dataset.make_initializable_iterator()
        inputs, labels = self.iterator.get_next()    
               
        '''
        hidden layer
        hidden layer weigths shape: input_length * num_hidden_node
        hidden layer bias shape: 1 * num_hidden_node
        '''
        self.hidden_layer_weights = tf.get_variable(name='hidden_layer_weights',
                                               shape=[inputs.shape[0],self.num_hidden_node],
                                               initializer=tf.truncated_normal_initializer(
                                                   mean=0.0,
                                                   stddev=0.01))
        self.hidden_layer_bias = tf.get_variable(name='hidden_layer_bias',
                                            shape=[1,self.num_hidden_node],
                                            initializer=tf.constant_initializer(0))
                                            
        hidden_layer_output = tf.tensordot(self.hidden_layer_weights, inputs, axes=(0,0))
        hidden_layer_output = tf.transpose(hidden_layer_output) + self.hidden_layer_bias
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

        correct_preds = tf.equal(tf.argmax(output_layer_output, 1), tf.argmax(labels,1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        '''
        loss
        '''
        self.loss = tf.losses.softmax_cross_entropy(labels,output_layer_output)

        '''optimizer'''
        
        #self.optimizer = tf.train.GradientDescentOptimizer(
            #learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
            
    def train(self):
        #Create a session and start to train and optimize weights and bias

        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            
            for i in range(self.epoch):
                total_loss = 0
                step = 0
                accurate = 0
                try:
                    while(True):
                        one_accurate,oneloss,_= sess.run([self.accuracy, self.loss, self.optimizer])
                        step += 1
                        total_loss += oneloss
                        accurate += one_accurate

                        if(step % 500 == 0):
                            print('Average loss at step {}, epoch {} : {:5.1f} / {:5.1f}'.
                                  format(step, i, total_loss/step, accurate/step))
 
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
 
                
            
                   
        
    def test(self):
        #Create a session and start to test
        pass
    
    def accuracy(self):
        pass
if __name__=='__main__':
    num_hidden_node = 100
    num_class = 10
    batch_size = 128
    epoch = 10
    learning_rate = 0.05
    nt = Networks(num_hidden_node,num_class, batch_size,epoch, learning_rate)
    nt.import_dataset()
    nt.build_graph()
    nt.train()
    
