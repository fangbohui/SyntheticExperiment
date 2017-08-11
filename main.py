from __future__ import print_function
import tensorflow as tf
import numpy as np
import pylab as pl
import random
import time
import os

tag = '' + time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
logdir = 'log/' + tag
os.system("mkdir " + logdir)
print ("dir = ", logdir)

train_size = 1000
test_size = 300
field_number = 100
field = np.random.random_integers(1, 20, field_number)
dimension = field.sum()

print("field:")
print(field)
print("dimension = ", dimension)
config_parameter = open(logdir + "/config.txt", 'w', buffering = -1)
config_parameter.write("train_size = %d\n" % train_size)
config_parameter.write("tets_size = %d\n" % test_size)
config_parameter.write("dimension = %d\n" % dimension)
config_parameter.write("field_number = %d\n" % field_number)
config_parameter.write("fields = [")
for i in range(field_number):
    config_parameter.write("%d, " % field[i])
config_parameter.write("]\n")


def make_parameters():
    file_parameter = open(logdir + "/parameter.txt", 'w', buffering = -1)
    # generate the probabilies
    # means the random_bound
    file_parameter.write("Bound-dimension = %d\n" % dimension)
    for i in range(dimension):
        #file_parameter.write("%.2f\n" % random.random())
        file_parameter.write("%.6f\n" % (random.random()))
    # generate the W
    # Wi * Xi
    file_parameter.write("W-dimension = %d\n" % dimension)
    for i in range(dimension):
        file_parameter.write("%.6f\n" % (random.random()))
        # file_parameter.write("0.0\n")
    # generate the V
    # Vij * Xi * Xj
    file_parameter.write("V-dimension = %d\n" % dimension)
    for i in range(dimension):
        for j in range(dimension):
            file_parameter.write("%.6f\n" % (random.random() - 0.5))
    file_parameter.close()


def sampling():
    # read the initial parameters and distributions
    file_parameter = open(logdir + "/parameter.txt", 'r', buffering = -1)
    train_X = np.zeros([train_size, dimension])
    test_X = np.zeros([test_size, dimension])
    prob = np.arange(dimension, dtype = float)
    W = np.zeros(dimension, dtype = float)
    V = np.zeros([dimension, dimension], dtype = float)
    file_parameter.readline()
    for i in range(dimension):
        s = file_parameter.readline().strip()
        prob[i] = float(s)
    file_parameter.readline()
    for i in range(dimension):
        W[i] = float(file_parameter.readline())
    file_parameter.readline()
    for i in range(dimension):
        for j in range(dimension):
            V[i][j] = float(file_parameter.readline())
    file_parameter.close()
    print("reading all the parameters")
    # sampling
    cnt = 0
    for i in range(field.size):
        sum = prob[cnt:cnt+field[i]].sum()
        prob[cnt:cnt+field[i]] /= sum
        cnt += field[i]

    cnt = 0
    for i in range(field_number):
        p = prob[cnt:cnt+field[i]]
        sample = np.random.choice(field[i], train_size, p = p)
        # train_X[:, sample + cnt] = 1
        for j in range(train_size):
            train_X[j][cnt + sample[j]] = 1
        sample = np.random.choice(field[i], test_size, p = p)
        # test_X[:, sample + cnt] = 1
        for j in range(test_size):
            test_X[j][cnt + sample[j]] = 1
        cnt += field[i]
    print("calculated all the train_X[i], test_X[i]")

    # y = sum_1^n w_i x_i + sum_1^n(sum_1^n vij * xi * xj)
    # batch * dim * 1
    p = np.expand_dims(train_X, 2)
    # batch * 1 * dim
    q = np.expand_dims(train_X, 1)
    # batch * 1
    train_Y = train_X.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())
    for i in range(train_size):
        train_Y[i] = np.random.normal(loc=train_Y[i], scale=0.1, size=None)
    sum = 0.0
    for i in range(train_size):
        sum += train_Y[i]
    bound = sum / train_size
    for i in range(train_size):
        if (train_Y[i] >= bound):
            train_Y[i] = 1
        else:
            train_Y[i] = 0

    p = np.expand_dims(test_X, 2)
    q = np.expand_dims(test_X, 1)
    test_Y = test_X.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())
    for i in range(test_size):
        test_Y[i] = np.random.normal(loc=test_Y[i], scale=0.1, size=None)
    for i in range(test_size):
        if (test_Y[i] >= bound):
            test_Y[i] = 1
        else:
            test_Y[i] = 0
    print("calculated all the train_Y[i], test_Y[i]")
    return train_X, train_Y, test_X, test_Y


def get_variable(init_type='tnormal', shape=None, name=None, minval=-0.1, maxval=0.1, mean=None, stddev=None, dtype=tf.float64):
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)


def train(tr_X, tr_Y, te_X, te_Y):
    learning_rate = 0.01
    training_epochs = 10000
    display_step = 10
    net_size = 128
    config_parameter.write("learning_rate = %.5f\n" % learning_rate)
    config_parameter.write("training epochs = %d\n" % training_epochs)
    config_parameter.write("display_step = %d\n" % display_step)
    config_parameter.write("net_size = %d\n" % net_size)
    config_parameter.write("layer_number = 2\n")

    train_X = np.asarray(tr_X).reshape(train_size, dimension)
    train_Y = np.asarray(tr_Y).reshape(train_size, 1)
    test_X = np.asarray(te_X).reshape(test_size, dimension)
    test_Y = np.asarray(te_Y).reshape(test_size, 1)

    X = tf.placeholder("float64", [None, dimension])
    Y = tf.placeholder("float64", [None, 1])

    W_h1 = get_variable('uniform', [dimension, net_size], 'w_h1')
    W_h2 = get_variable('uniform', [net_size, net_size], 'w_h2')
    W_o = get_variable('uniform', [net_size, 1], 'w_o')

    Bias_h1 = get_variable('zero', [net_size], "Bias_h1")
    Bias_h2 = get_variable('zero', [net_size], "Bias_h2")
    Bias_o = get_variable('zero', [1], "Bias_o")
    # W_h3 = get_xavier([dimension, dimension], 'w_h3')
    # Bias_h3 = tf.Variable(tf.zeros(shape = [dimension], dtype = tf.float64), name = "Bias_h3")
    # W_h4 = get_xavier([dimension, dimension], 'w_h4')
    # Bias_h4 = tf.Variable(tf.zeros(shape = [dimension], dtype = tf.float64), name = "Bias_h4")

    l1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W_h1), Bias_h1))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, W_h2), Bias_h2))
    # l = tf.tanh(tf.add(tf.matmul(l, W_h3), Bias_h3))
    # l = tf.tanh(tf.add(tf.matmul(l, W_h4), Bias_h4))
    output = tf.add(tf.matmul(l1, W_o), Bias_o)

    pred = tf.nn.sigmoid(output)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=pred))
    auc = tf.metrics.auc(predictions=pred, labels=Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    epoch_vector = np.zeros([training_epochs / display_step + 1])
    AUC_vector = np.zeros([training_epochs / display_step + 1])

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())
        # Fit all training data
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                train_auc = sess.run(auc, feed_dict={X: train_X, Y: train_Y})[1]
                training_cost = sess.run(cross_entropy, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", '%04d' % (epoch + 1), "Training loss =", "%.9f" % training_cost.sum(), "AUC=", train_auc, "Bias_o=", sess.run(Bias_o))
                testing_cost = sess.run(cross_entropy, feed_dict={X: test_X, Y: test_Y})
                test_auc = sess.run(auc, feed_dict={X: test_X, Y: test_Y})[1]
                print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", test_auc)
                epoch_vector[(epoch + 1) / display_step] = epoch
                AUC_vector[(epoch + 1) / display_step] = test_auc

        print("\nOptimization Finished!")
        train_auc = sess.run(auc, feed_dict={X: train_X, Y: train_Y})[1]
        training_cost = sess.run(cross_entropy, feed_dict={X: train_X, Y: train_Y})
        print("Training loss =", "%.9f" % training_cost.sum(), "AUC=", (train_auc), "Bias_o=", sess.run(Bias_o))
        # Testing example, as requested (Issue #2)

        print("Testing... (Cross Entropy)")
        testing_cost = sess.run(cross_entropy, feed_dict={X: test_X, Y: test_Y})
        test_auc = sess.run(auc, feed_dict={X: test_X, Y: test_Y})[1]
        print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", (test_auc))
        print("Absolute Cost difference:", abs(training_cost.sum() - testing_cost.sum()))

        # print(test_Y)
        # print(sess.run(pred, feed_dict={X:test_X}))

    print (epoch_vector)
    print (AUC_vector)
    pl.plot(epoch_vector, AUC_vector)
    pl.xlabel("epoch")
    pl.ylabel("AUC")
    pl.savefig(logdir + "/curve.png")
    pl.show()


make_parameters()
train_X, train_Y, test_X, test_Y = sampling()
train(tr_X = train_X, tr_Y = train_Y, te_X = test_X, te_Y = test_Y)
