import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#network structure: input layer --->
#                   one convNet --->
#           one fully connected --->
#                  output layer

#model hyperparameter
learning_rate = 0.01
conv_filter_size = 3
conv_num_filters = 64
fc_hidden_node = 100
num_classes = 10
batch_size = 128
epoch = 1

#import data (input layer)
mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
x_train = mnist.train.images
x_train = np.reshape(x_train,[-1, 28, 28,1])
y_train = mnist.train.labels
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

x_test = mnist.test.images
x_test = np.reshape(x_test,[-1, 28, 28,1])
y_test = mnist.test.labels
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

img, labels = iterator.get_next()


#build the first convolutional layer 
conv_weights = tf.get_variable(name='conv_weights',
                               shape=[conv_filter_size, conv_filter_size,1,conv_num_filters],
                               initializer=tf.truncated_normal_initializer(0,0.01)
                               )
conv_bias = tf.get_variable(name='conv_bias',
                            shape=[conv_num_filters],
                            initializer=tf.constant_initializer(0.0)
                            )
conv_output = tf.nn.conv2d(img,conv_weights,strides=[1,1,1,1], padding='SAME')
conv_output = tf.nn.bias_add(conv_output,conv_bias)
conv_output = tf.nn.relu(conv_output)

#build fully connected layer
#flatten conv_output
conv_output_len = conv_output[1].shape[0]*conv_output[1].shape[1]*conv_output[1].shape[2]
hidden_layer_input = tf.reshape(conv_output,[-1,conv_output_len])
hidden_layer_weights = tf.get_variable(name='fc_weights',
                                       shape=[conv_output_len, fc_hidden_node],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
hidden_layer_bias = tf.get_variable(name='fc_bias',
                                    shape=[1,fc_hidden_node],
                                    initializer=tf.constant_initializer(0.0))
hidden_layer_output = tf.matmul(hidden_layer_input, hidden_layer_weights)
hidden_layer_output = tf.add(hidden_layer_output, hidden_layer_bias)
hidden_layer_output = tf.nn.relu(hidden_layer_output)

#build output layer
output_layer_weights = tf.get_variable(name='output_weights',
                                       shape=[fc_hidden_node,num_classes],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

output_layer_bias = tf.get_variable(name='output_bias',
                                    shape=[1, num_classes],
                                    initializer=tf.constant_initializer(0.0))
logit = tf.matmul(hidden_layer_output, output_layer_weights)
logit = tf.add(logit,output_layer_bias)

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logit)
loss = tf.reduce_sum(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_preds = tf.equal(tf.argmax(logit,1), tf.argmax(labels,1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    #start train
    for i in range(epoch):
        sess.run(train_init)
        total_loss = 0
        n_batch = 0
        try:
            while True:
                _, batch_loss = sess.run([optimizer,loss])
                total_loss += batch_loss
                n_batch += 1
                print('Average loss in epoch {}, batch {}:{:5.2f}'.format(i, n_batch, total_loss/n_batch)) 
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss in epoch {}:{:5.2f}'.format(i, total_loss/n_batch))


    #start test
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            batch_correct_preds = sess.run(accuracy)
            total_correct_preds += batch_correct_preds
    except tf.errors.OutOfRangeError:
        pass
    print('Accuracy on test data: {:5.3f}'.format(total_correct_preds/10000))
          
            
            
    
        
        

