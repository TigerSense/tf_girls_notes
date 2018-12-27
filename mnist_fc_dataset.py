import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


#define model hyperparameter
num_hidden_node = 100
num_class = 10
learning_rate = 0.01
epoch = 10
batch_size = 128


mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
n_test = x_test.shape[0]
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

#hidden layer
hidden_layer_weights = tf.get_variable(name='hidden_layer_weights',
                                       shape=[img.shape[1], num_hidden_node],
                                       initializer=tf.truncated_normal_initializer(
                                           mean=0.0,
                                           stddev=0.1)
                                       )
hidden_layer_bias = tf.get_variable(name='hidden_layer_bias',
                                    shape=[1, num_hidden_node],
                                    initializer=tf.constant_initializer(0.0))

hidden_layer_output = tf.matmul(img,hidden_layer_weights) + hidden_layer_bias
hidden_layer_output = tf.nn.relu(hidden_layer_output)

#output layer
output_layer_weights = tf.get_variable(name='output_layer_weights',
                                       shape=[hidden_layer_output.shape[1],num_class],
                                       initializer=tf.truncated_normal_initializer(
                                           mean=0.0,
                                           stddev=0.01
                                           )
                                       )
output_layer_bias = tf.get_variable(name='output_layer_bias',
                                    shape=[1,num_class],
                                    initializer=tf.constant_initializer(0.0))
logit = tf.matmul(hidden_layer_output,output_layer_weights) + output_layer_bias

#entroy and loss
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=label,name='entropy')

loss = tf.reduce_mean(entropy,name='loss')

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#compute prediction accuracy
correct_preds = tf.equal(tf.argmax(logit,1), tf.argmax(label,1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        sess.run(train_init)
        n_batches = 0
        total_loss = 0
        try:
            while True:
                _, epoch_loss = sess.run([optimizer, loss])
                n_batches += 1
                total_loss += epoch_loss
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch{}, batches {}:{:5.2f}'.format(i, n_batches, total_loss/n_batches))

    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            batch_accuracy = sess.run(accuracy)
            total_correct_preds += batch_accuracy
    except tf.errors.OutOfRangeError:
        pass
    print('Accuracy {0}'.format(total_correct_preds/n_test))
    
        
                                           
                                       

 
