from __future__ import print_function
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
import pylab as auc_curve
import random
import time
import os

tag = '' + time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
logdir = 'log/' + tag
os.system("mkdir " + logdir)
print("dir = ", logdir)

train_size = 1000
test_size = 300
field_number = 35
y_scale = 10000
field = np.random.random_integers(1, 40, field_number)
dimension = field.sum()
network = np.array([dimension, 128, 128, 1])

learning_rate = 0.1
training_epochs = 1000
display_step = 10

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
        file_parameter.write("0.0\n")
    file_parameter.write("V-dimension = %d\n" % dimension)
    V = np.zeros([dimension, dimension])
    for i in range(dimension):
        for j in range(dimension):
            V[i][j] = (random.random() - 0.5) * 1000
            file_parameter.write("%.6f\n" % V[i][j])
    file_parameter.close()
    print ("the stddev of V-distribution is %.5f" % np.std(V))


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
    print ("y-std = %.5f" % np.std(y))
    for i in range(train_size):
        y[i] = np.random.normal(loc=y[i], scale=y_scale, size=None)
    sum = 0.0
    for i in range(train_size):
        sum += y[i]
    global bound
    bound = 1.0 * sum / train_size


def sampling(size):
    x = np.zeros([size, dimension])
    x_indices = np.zeros([size * field_number, 2], dtype=int)
    cnt = 0
    tmp = 0
    for i in range(field_number):
        p = prob[cnt:cnt+field[i]]
        sample = np.random.choice(field[i], size, p = p)
        for j in range(size):
            x[j][cnt + sample[j]] = 1
            x_indices[tmp][0] = j
            x_indices[tmp][1] = cnt + sample[j]
            tmp += 1
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
    y = np.asarray(y).reshape(size, 1)
    xx = tf.SparseTensorValue(x_indices, np.ones([size * field_number]), [size, dimension])
    return xx, y


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
    config_parameter.write("learning_rate = %.5f\n" % learning_rate)
    config_parameter.write("training epochs = %d\n" % training_epochs)
    config_parameter.write("display_step = %d\n" % display_step)
    config_parameter.write("hidden_layer_number = %d\n" % (network.size - 2))

    X = tf.sparse_placeholder("float64", [None, dimension])
    Y = tf.placeholder("float64", [None, 1])

    W_h = []
    Bias_h = []
    l = []
    for i in range(1, network.size):
        W_h.append(get_variable('xavier', [network[i - 1], network[i]], "w_h" + str(i)))
        Bias_h.append(get_variable('zero', [network[i]], "Bias_h" + str(i)))
    for i in range(network.size - 1):
        if (i == 0):
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

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())
        for epoch in range(training_epochs):
            x, y = sampling(train_size)
            sess.run(optimizer, feed_dict={X: x, Y: y})
            if (epoch + 1) % display_step == 0:
                prediction = sess.run(pred, feed_dict={X:x})
                train_auc = roc_auc_score(y.flatten(), prediction.flatten())
                training_cost = sess.run(cross_entropy, feed_dict={X: x, Y: y})
                print("Epoch:", '%04d' % (epoch + 1), "Training loss =", "%.9f" % training_cost.sum(), "AUC=", train_auc, "Bias_o=", sess.run(Bias_h[network.size - 2]))
                x, y = sampling(test_size)
                testing_cost = sess.run(cross_entropy, feed_dict={X: x, Y: y})
                prediction = sess.run(pred, feed_dict={X: x})
                test_auc = roc_auc_score(y.flatten(), prediction.flatten())
                print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", test_auc)
                epoch_vector[(epoch + 1) / display_step] = epoch
                AUC_vector[(epoch + 1) / display_step] = test_auc

        print("\nOptimization Finished!")
        x, y = sampling(train_size)
        prediction = sess.run(pred, feed_dict={X:x})
        train_auc = roc_auc_score(y.flatten(), prediction.flatten())
        training_cost = sess.run(cross_entropy, feed_dict={X: x, Y: y})
        print("Training loss =", "%.9f" % training_cost.sum(), "AUC=", (train_auc), "Bias_o=", sess.run(Bias_h[network.size - 2]))

        print("Testing... (Cross Entropy)")
        x, y = sampling(test_size)
        testing_cost = sess.run(cross_entropy, feed_dict={X: x, Y: y})
        prediction = sess.run(pred, feed_dict={X:x})
        test_auc = roc_auc_score(y.flatten(), prediction.flatten())
        print("Testing loss = ", "%.9f" % (testing_cost.sum()), "AUC=", (test_auc))
        print("Absolute Loss Difference:", abs(training_cost.sum() - testing_cost.sum()))
        config_parameter.write("Testing loss = %.9f\n" % (testing_cost.sum()))
        config_parameter.write("AUC = %.9f\n" % (test_auc))
        avg_auc = 0
        cnt = 0
        reversed_auc = AUC_vector[::-1]
        for i in range(reversed_auc.size):
            avg_auc += reversed_auc[i]
            cnt +=1
            if (cnt == 100):
                break
        avg_auc /= cnt
        config_parameter.write("AUC-AVG = %.9f\n" % avg_auc)
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
train()
