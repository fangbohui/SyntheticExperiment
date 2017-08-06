from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
rng = np.random


train_size = 10000
test_size = 1000
dimension = 6
#field_number = 5
#field = np.array([2, 4, 5, 7, 12])
field_number = 3
field = np.array([2, 2, 2])


def make_parameters():
    file_parameter = open("parameter.txt", 'w', buffering = -1)
    # generate the probabilies
    # means the random_bound
    file_parameter.write("Bound-dimension = %d\n" % dimension)
    for i in range(dimension):
        #file_parameter.write("%.2f\n" % random.random())
        file_parameter.write("%.6f\n" % (((random.random())) * 10000))
    # generate the W
    # Wi * Xi
    file_parameter.write("W-dimension = %d\n" % dimension)
    for i in range(dimension):
        file_parameter.write("%.6f\n" % (((random.random())) * 10000))
    # generate the V
    # Vij * Xi * Xj
    file_parameter.write("V-dimension = %d\n" % dimension)
    for i in range(dimension):
        for j in range(dimension):
            if (i == 1 and j == 3):
                file_parameter.write("%.6f\n" % (((random.random() - 0.5)) * 10000))
            elif (i == 1 and j == 5):
                file_parameter.write("%.6f\n" % (((random.random() - 0.5)) * 10000))
            elif (i == 3 and j == 5):
                file_parameter.write("%.6f\n" % (((random.random() - 0.5)) * 10000))
            else:
                file_parameter.write("0.00\n")
    file_parameter.close()


def sampling():
    # read the initial parameters and distributions
    file_parameter = open("parameter.txt", 'r', buffering = -1)
    train_X = np.zeros([train_size, dimension])
    train_Y = np.zeros(train_size)
    test_X = np.zeros([test_size, dimension])
    test_Y = np.zeros(test_size)
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


    # print the sampling probabilities
    cnt = 0
    for i in range(field_number):
        for j in range(field[i]):
            print(prob[cnt + j], end = ' ')
        cnt += field[i]
        print("")


    cnt = 0
    print("begin to calc all the train_X[i], test_X[i]")
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

    # y = sum_1^n w_i x_i + sum_1^n(sum_1^n vij * xi * xj)
    # batch * dim * 1
    p = np.expand_dims(train_X, 2)
    # batch * 1 * dim
    q = np.expand_dims(train_X, 1)
    # batch * 1
    train_Y = train_X.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())

    p = np.expand_dims(test_X, 2)
    q = np.expand_dims(test_X, 1)
    test_Y = test_X.dot(W) + (p * q).reshape([-1, dimension**2]).dot(V.flatten())

    return train_X, train_Y, test_X, test_Y


def train(tr_X, tr_Y, te_X, te_Y):
    learning_rate = 3
    training_epochs = 10000
    batch_size = 1
    display_step = 1

    train_X = np.asarray(tr_X).reshape(train_size, dimension)
    train_Y = np.asarray(tr_Y).reshape(train_size, 1)
    test_X = np.asarray(te_X).reshape(test_size, dimension)
    test_Y = np.asarray(te_Y).reshape(test_size, 1)

    X = tf.placeholder("float64", [None, dimension])
    Y = tf.placeholder("float64", [None, 1])

    W_h1 = tf.Variable(tf.random_uniform(shape = [dimension, dimension], minval = -0.1, maxval = 0.1, dtype = tf.float64), name = "W_h1")
    Bias_h1 = tf.Variable(tf.zeros(shape = [dimension], dtype = tf.float64), name = "Bias_h1")
    W_h2 = tf.Variable(tf.random_uniform(shape = [dimension, dimension], minval = -0.1, maxval = 0.1, dtype = tf.float64), name = "W_h2")
    Bias_h2 = tf.Variable(tf.zeros(shape = [dimension], dtype = tf.float64), name = "Bias_h2")
    W_o = tf.Variable(tf.random_uniform(shape = [dimension, dimension], minval = -0.1, maxval = 0.1, dtype = tf.float64), name = "W_o")
    Bias_o = tf.Variable(tf.zeros(shape = [1], dtype = tf.float64), name = "Bias_o")

    l = tf.tanh(tf.add(tf.matmul(X, W_h1), Bias_h1))
    l = tf.tanh(tf.add(tf.matmul(l, W_h2), Bias_h2))
    l = tf.add(tf.matmul(l, W_o), Bias_o)

    pred = l
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, Y))))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            offset = (epoch * batch_size) % (train_size - batch_size)
            batch_X = train_X[offset:(offset + batch_size), :]
            batch_Y = train_Y[offset:(offset + batch_size), :]
            sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", '%04d' % (epoch + 1), "Training RMSE=", "{:.9f}".format(c), "Bias_o=", sess.run(Bias_o))
                #print("W_o=", sess.run(W_o))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training RMSE=", training_cost, "Bias_o=", sess.run(Bias_o), '\n')

        # Testing example, as requested (Issue #2)

        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(pred, Y)))), feed_dict={X: test_X, Y: test_Y})
        print("Testing RMSE=", testing_cost)
        print("Absolute RMSE difference:", abs(training_cost - testing_cost))



make_parameters()
train_X, train_Y, test_X, test_Y = sampling()
train(tr_X = train_X, tr_Y = train_Y, te_X = test_X, te_Y = test_Y)
