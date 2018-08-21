import numpy as np
import tensorflow as tf
#import cifar_10_data

#import os
#os.chdir(r'C:\Users\billy\source\repos\cifar-10\cifar-10')

n_conv1,n_conv2,n_conv3,n_conv4,n_conv5,n_full1,n_full2=(64,64,128,128,128,384,192)
_RESHAPE_SIZE=4*4*128
n_classes=10
learning_rate=1e-5
n_epochs = 1000
batch_size = 100
l1_reg=0.0
keep_prob=0.6
_SAVE_PATH="./tensorboard/cifar-10/noreg_withdrop0.6/"


'''
plt.imshow(data[8])
plt.show()
'''

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
X=tf.placeholder(tf.float32,shape=[None,32,32,3],name='image')
X_drop=tf.contrib.layers.dropout(X,keep_prob,is_training=is_training)
y=tf.placeholder(tf.int32,name='y')

with tf.name_scope("cnn"):
    with tf.contrib.slim.arg_scope([tf.contrib.layers.conv2d,tf.contrib.layers.fully_connected],
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   #weights_initializer=tf.contrib.layers.variance_scaling_initializer(),#容易爆炸
                                   activation_fn=tf.nn.leaky_relu,#leaky_relu收敛似乎更快
                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                   normalizer_params={'is_training': is_training,'updates_collections': None}):
                                   #weights_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-5)):
        conv1=tf.contrib.layers.conv2d(inputs=X,num_outputs=n_conv1, kernel_size=5,stride=1,
                                       padding='SAME',scope="conv_1")
        pool1=tf.contrib.layers.max_pool2d(inputs=conv1,kernel_size=3,stride=2,padding='SAME',scope="pool_1")
        pool1_drop=tf.contrib.layers.dropout(pool1,keep_prob,is_training=is_training)
        conv2=tf.contrib.layers.conv2d(inputs=pool1_drop,num_outputs=n_conv2, kernel_size=5,stride=1,
                                       padding='SAME', scope="conv_2")
        pool2=tf.contrib.layers.max_pool2d(inputs=conv2,kernel_size=3,stride=2,padding='SAME',scope="pool_2")
        pool2_drop=tf.contrib.layers.dropout(pool2,keep_prob,is_training=is_training)
        conv3=tf.contrib.layers.conv2d(inputs=pool2_drop,num_outputs=n_conv3, kernel_size=3,stride=1,
                                       padding='SAME', scope="conv_3")
        conv4=tf.contrib.layers.conv2d(inputs=conv3,num_outputs=n_conv4, kernel_size=3,stride=1,
                                       padding='SAME', scope="conv_4")
        conv5=tf.contrib.layers.conv2d(inputs=conv4,num_outputs=n_conv5, kernel_size=3,stride=1,
                                       padding='SAME', scope="conv_5")
        pool3=tf.contrib.layers.max_pool2d(inputs=conv5,kernel_size=3,stride=2,padding='SAME',scope="pool_3")
        pool3_flatten=tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        pool3_drop=tf.contrib.layers.dropout(pool3_flatten,keep_prob,is_training=is_training)
        fully_connected1=tf.contrib.layers.fully_connected(inputs=pool3_drop, num_outputs=n_full1,
                                                           scope="fully_connected_1")
        fully_connected1_drop=tf.contrib.layers.dropout(fully_connected1,keep_prob,is_training=is_training)
        fully_connected2=tf.contrib.layers.fully_connected(inputs=fully_connected1_drop, num_outputs=n_full2,
                                                           scope="fully_connected_2")
        logits = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_classes,
                                                   activation_fn=None,normalizer_fn=None,scope="outputs")

with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss=tf.reduce_mean(xentropy,name="base_loss")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss]+reg_losses, name="loss")    

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    training_op = optimizer.minimize(loss,global_step=global_step)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, k=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
Train_acc_summary=tf.summary.scalar('Train_Accuracy', accuracy)
Test_acc_summary=tf.summary.scalar('Test_Accuracy', accuracy)

file_writer = tf.summary.FileWriter(_SAVE_PATH, tf.get_default_graph())