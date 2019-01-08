import tensorflow as tf
import numpy as np
import os

import utils
import word2vec_utils
from tensorflow.contrib.tensorboard.plugins import projector

#Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
LEARNING_RATE = 1.0
SKIP_WINDOW = 1
NUM_SAMPLED = 64
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

#Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000

class SkipGramModel:
    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.get_variable(name='global_step',initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP

    def __import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()
            
    def __create_embedding(self):
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable(name='embed_matrix',
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix,self.center_words,name='embedding')

    def __create_loss(self):
        with tf.name_scope('loss'):
            nce_weight = tf.get_variable(name='nce_weight',
                                         shape=[self.vocab_size, self.embed_size],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0/(self.embed_size**0.5)))
            nce_bias = tf.get_variable(name='nce_bias',
                                       initializer=tf.zeros([self.vocab_size]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                      biases=nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size),name='loss')
    def __create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step)

    def __create_summaries(self):
        tf.summary.scalar('loss',self.loss)
        tf.summary.histogram('histogram loss',self.loss)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self.__import_data()
        self.__create_embedding()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_summaries()

    def train(self,num_train_steps):
        saver = tf.train.Saver()
        initial_step = 0
        utils.safe_mkdir('checkpoints')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
            total_loss = 0
            writer = tf.summary.FileWriter('graphs/word2vec/lr'+str(self.learning_rate), sess.graph)
            inital_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary,global_step=index)
                    total_loss += loss_batch
                    if (index+1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0
                        saver.save(sess,'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    def visualize(self, visual_fld, num_visualize):
        word2vec_utils.most_common_words(visual_fld,num_visualize)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
            final_embed_matrix = sess.run(self.embed_matrix)
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            # create the embedding projectorc
            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # specify where to find the meta data
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld,'model.ckpt'),1)
            

                
    
def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL,EXPECTED_BYTES, VOCAB_SIZE,BATCH_SIZE,SKIP_WINDOW,VISUAL_FLD)

def main():
    print('load dataset')
    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE,1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE,BATCH_SIZE,NUM_SAMPLED,LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VISUAL_FLD, NUM_VISUALIZE)

    
if __name__=='__main__':
    main()
