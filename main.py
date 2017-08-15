from __future__ import print_function
import tensorflow as tf
import numpy as np
import pylab as auc_curve
import random
import time
import os
from sklearn.metrics import roc_auc_score

tag = '' + time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
logdir = 'log/' + tag
os.system("mkdir " + logdir)
print("dir = ", logdir)

train_size = 1000
test_size = 300
field_number = 20
y_scale = 0
field = np.random.random_integers(1, 20, field_number)
dimension = field.sum()
network = np.array([dimension, 64, 1])

from scipy.stats import ttest_ind

print("field:")
print(field)
print("dimension = ", dimension)
config_parameter = open(logdir + "/config.txt", 'w', buffering = -1)
config_parameter.write("train_size = %d\n" % train_size)
config_parameter.write("tets_size = %d\n" % test_size)
config_parameter.write("dimension = %d\n" % dimension)
config_parameter.write("y distribution_scale = %.5f\n" % y_scale)
config_parameter.write("field_number = %d\n" % field_number)
config_parameter.write("fields = [")
for i in range(field_number):
    config_parameter.write("%d, " % field[i])
config_parameter.write("]\n")
config_parameter.write("network = [")
for i in range(network.size):
    config_parameter.write("%d, " % network[i])
config_parameter.write("]\n")


def make_parameters():
    file_parameter = open(logdir + "/parameter.txt", 'w', buffering = -1)
    file_parameter.write("Bound-dimension = %d\n" % dimension)
    for i in range(dimension):
        file_parameter.write("%.6f\n" % (random.random()))
    file_parameter.write("W-dimension = %d\n" % dimension)
    for i in range(dimension):
        # file_parameter.write("%.6f\n" % (random.random() / 10))
        file_parameter.write("0.0\n")
    file_parameter.write("V-dimension = %d\n" % dimension)
    V = np.zeros([dimension, dimension])
    for i in range(dimension):
        for j in range(dimension):
            #if (np.random.random() < 0.):
            V[i][j] = (random.random() - 0.5) * 1000
            file_parameter.write("%.6f\n" % V[i][j])
            # else:
            #     file_parameter.write("0.0\n")
    file_parameter.close()
    print (np.std(V))


prob = np.arange(dimension, dtype = float)
W = np.zeros(dimension, dtype = float)
V = np.zeros([dimension, dimension], dtype = float)
bound = 0

def get_parameters():
    # read the initial parameters and distributions
    file_parameter = open(logdir + "/parameter.txt", 'r', buffering = -1)
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

    cnt = 0
    for i in range(field.size):
        sum = prob[cnt:cnt+field[i]].sum()
        prob[cnt:cnt+field[i]] /= sum
        cnt += field[i]

    x = np.zeros([train_size, dimension])
    cnt = 0
    for i in range(field_number):
        p = prob[cnt:cnt+field[i]]
        sample = np.random.choice(field[i], train_size, p = p)
        for j in range(train_size):
            x[j][cnt + sample[j]] = 1
        cnt += field[i]
    print("calculated all the train_X[i], test_X[i]")

    p = np.expand_dims(x, 2)
    q = np.expand_dims(x, 1)
    y = x.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())
    for i in range(train_size):
        y[i] = np.random.normal(loc=y[i], scale=y_scale, size=None)
    sum = 0.0
    for i in range(train_size):
        sum += y[i]
    global bound
    bound = 1.0 * sum / train_size


def sampling(size):
    x = np.zeros([size, dimension])
    x_indices = np.zeros([size, field_number], dtype = int)
    cnt = 0
    for i in range(field_number):
        p = prob[cnt:cnt+field[i]]
        sample = np.random.choice(field[i], size, p = p)
        for j in range(size):
            x[j][cnt + sample[j]] = 1
            x_indices[j][i] = cnt + sample[j]
        cnt += field[i]

    p = np.expand_dims(x, 2)
    q = np.expand_dims(x, 1)
    y = x.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())
    for i in range(size):
        y[i] = np.random.normal(loc=y[i], scale=y_scale, size=None)
    for i in range(size):
        if (y[i] >= bound):
            y[i] = 1
        else:
            y[i] = 0
    # p_values = []
    # for i in range(dimension):
    #     _, p_value = ttest_ind(train_X[:, i], test_X[:, i])
    #     p_values.append(p_value)
    #     print(p_value)
    # print(np.nanmean(p_values))
    # exit(0)
    # return train_X, train_Y, test_X, test_Y
    # y = np.asarray(y).reshape([size, 1])
    # x_indices.sort()
    return x_indices, y


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


def train():
    learning_rate = 0.1
    training_epochs = 600
    display_step = 10
    config_parameter.write("learning_rate = %.5f\n" % learning_rate)
    config_parameter.write("training epochs = %d\n" % training_epochs)
    config_parameter.write("display_step = %d\n" % display_step)
    config_parameter.write("layer_number = 2\n")

    # train_X = np.asarray(tr_X).reshape(train_size, dimension)
    tr_X = tf.placeholder("float64", [train_size, 2])
    tr_Y = tf.placeholder("float64", [train_size, 1])
    te_X = tf.placeholder("float64", [test_size, 2])
    te_Y = tf.placeholder("float64", [test_size, 1])
    train_X = tf.SparseTensorValue(tr_X, np.ones([train_size * field_number]), [train_size, dimension])
    test_X = tf.SparseTensorValue(te_X, np.ones([test_size * field_number]), [test_size, dimension])
    # train_Y = np.asarray(tr_Y).reshape()
    # test_X = np.asarray(te_X).reshape(test_size, dimension)
    # test_Y = np.asarray(te_Y)

    # X = tf.placeholder("float64", [None, dimension])
    # Y = tf.placeholder("float64", [None, 1])
    X = tf.sparse_placeholder("float64", [None, dimension])
    # X = tf.placeholder("int32", [None, field_number])
    Y = tf.placeholder("float64", [None, 1])

    W_h = []
    Bias_h = []
    l = []
    for i in range(1, network.size):
        W_h.append(get_variable('xavier', [network[i - 1], network[i]], "w_h" + str(i)))
        Bias_h.append(get_variable('zero', [network[i]], "Bias_h" + str(i)))
    for i in range(network.size - 1):
        if (i == 0):
            # W_h[0] = tf.gather(W_h[0], train_x_indices)
            l.append(tf.nn.tanh(tf.add(tf.sparse_tensor_dense_matmul(X, W_h[0]), Bias_h[0])))
        else:
            l.append(tf.nn.tanh(tf.add(tf.matmul(l[i - 1], W_h[i]), Bias_h[i])))
    output = l[network.size - 2]
    pred = tf.nn.sigmoid(output)

    l2 = 0.02
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)) + \
                    l2 * (sum(map(lambda x: tf.nn.l2_loss(x), W_h)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
    init = tf.global_variables_initializer()

    epoch_vector = np.zeros([training_epochs / display_step + 1])
    AUC_vector = np.zeros([training_epochs / display_step + 1])

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())
        # Fit all training data
        for epoch in range(training_epochs):
            x_indices, y = sampling(train_size)
            y = np.asarray(y).reshape(train_size, 1)
            print (x_indices)
            x = sess.run(train_X, feed_dict={tr_X:x_indices})
            return
            sess.run(optimizer, feed_dict={X: train_X, tr_Y: y})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                prediction = sess.run(pred, feed_dict={tr_X:x_indices})
                train_auc = roc_auc_score(y.flatten(), prediction.flatten())
                training_cost = sess.run(cross_entropy, feed_dict={tr_X: x_indices, tr_Y: y})
                print("Epoch:", '%04d' % (epoch + 1), "Training loss =", "%.9f" % training_cost.sum(), "AUC=", train_auc, "Bias_o=", sess.run(Bias_h[network.size - 2]))
                x_indices, y = sampling(test_size)
                testing_cost = sess.run(cross_entropy, feed_dict={te_X: x_indices, te_Y: y})
                prediction = sess.run(pred, feed_dict={te_X: x_indices})
                test_auc = roc_auc_score(y.flatten(), prediction.flatten())
                print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", test_auc)
                epoch_vector[(epoch + 1) / display_step] = epoch
                AUC_vector[(epoch + 1) / display_step] = test_auc

        print("\nOptimization Finished!")
        prediction = sess.run(pred, feed_dict={X:train_X})
        train_auc = roc_auc_score(train_Y.flatten(), prediction.flatten())
        training_cost = sess.run(cross_entropy, feed_dict={X: train_X, Y: train_Y})
        print("Training loss =", "%.9f" % training_cost.sum(), "AUC=", (train_auc), "Bias_o=", sess.run(Bias_h[network.size - 2]))
        # Testing example, as requested (Issue #2)

        print("Testing... (Cross Entropy)")
        testing_cost = sess.run(cross_entropy, feed_dict={X: test_X, Y: test_Y})
        prediction = sess.run(pred, feed_dict={X:test_X})
        train_auc = roc_auc_score(test_Y.flatten(), prediction.flatten())
        print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", (test_auc))
        print("Absolute Loss Difference:", abs(training_cost.sum() - testing_cost.sum()))
        config_parameter.write("Testing loss = %.9f\n" % (testing_cost.sum()))
        config_parameter.write("AUC = %.9f\n" % (test_auc))
        config_parameter.close()

    print (epoch_vector)
    print (AUC_vector)
    auc_curve.plot(epoch_vector, AUC_vector)
    auc_curve.xlabel("epoch")
    auc_curve.ylabel("AUC")
    auc_curve.savefig(logdir + "/AUC-curve.png")
    auc_curve.show()


make_parameters()
get_parameters()
# train_X, train_Y, test_X, test_Y = sampling()
train()
