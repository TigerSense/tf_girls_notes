import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
import matplotlib.pyplot as plt

def vector_to_matrix_mnist(data):
    return np.reshape(data,[-1, 28, 28])

def invert_grayscale(data):
    return 1-data

def create_sprite_image(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    spriteimage = np.ones([n_plots*img_h, n_plots*img_w])
    for i in range(n_plots):
        for j in range(n_plots):
            if i*n_plots+j<images.shape[0]:
                spriteimage[i*img_h:(i+1)*img_h,j*img_w:(j+1)*img_w] = images[i*n_plots+j]
    return spriteimage

LOG_DIR = 'minimalsample'
TO_EMBED_COUNT = 500
NAME_TO_VISUALIZE_VARIABLE = 'mnistembedding'
path_to_mnist_sprites = 'mnistdigits.png'
path_to_mnist_metadata = 'metadata.tsv'

mnist = input_data.read_data_sets('mnist_data/',one_hot=False)
batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

#create sprite image
to_visualize = batch_xs
to_visualize = vector_to_matrix_mnist(to_visualize)
to_visualize = invert_grayscale(to_visualize)
sprite_image = create_sprite_image(to_visualize)

plt.imsave(path_to_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image)

#create meta data
with open(path_to_mnist_metadata,'w') as f:
    f.write('Index\tLable\n')
    for index, label in enumerate(batch_ys):
        f.write("%d\t%d\n" % (index,label))


embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALIZE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = path_to_mnist_metadata
embedding.sprite.image_path = path_to_mnist_sprites

embedding.sprite.single_image_dim.extend([28,28])

projector.visualize_embeddings(summary_writer,config)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR,"model.ckpt"),1)



