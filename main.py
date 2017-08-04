from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
rng = np.random


train_size = 100
test_size = 10
dimension = 30


def make_parameters():
    file_parameter = open("parameter.txt", 'w', buffering = -1)
    # generate the probabilies
    # means the random_bound
    file_parameter.write("Bound-dimension = %d\n" % dimension)
    for i in range(dimension):
        file_parameter.write("%.2f\n" % random.random())
    # generate the W
    # Wi * Xi
    file_parameter.write("W-dimension = %d\n" % dimension)
    for i in range(dimension):
        file_parameter.write("%.2f\n" % random.random())
    # generate the V
    # Vij * Xi * Xj
    file_parameter.write("V-dimension = %d\n" % dimension)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                file_parameter.write("0.00\n")
            else:
                file_parameter.write("%.2f\n" % random.random())
    file_parameter.close()


def sampling():
    # read the initial parameters and distributions
    file_parameter = open("parameter.txt", 'r', buffering = -1)
    train_X = np.zeros(train_size * dimension, dtype = float).reshape(train_size, dimension)
    train_y = np.zeros(train_size, dtype = float)
    test_X = np.zeros(test_size * dimension, dtype = float).reshape(test_size, dimension)
    test_y = np.zeros(test_size, dtype = float)
    prob = np.arange(dimension, dtype = float)
    W = np.zeros(dimension, dtype = float)
    V = np.zeros(dimension * dimension, dtype = float).reshape(dimension, dimension)
    file_parameter.readline()
    for i in range(dimension):
        s = file_parameter.readline().replace('\n', '')
        prob[i] = float(s)
    file_parameter.readline()
    for i in range(dimension):
        W[i] = float(file_parameter.readline())
    file_parameter.readline()
    for i in range(dimension):
        for j in range(dimension):
            V[i][j] = float(file_parameter.readline())
    file_parameter.close()
    print("readin all the parameters")
    # sampling
    field_number = 5
    field = np.array([2, 4, 5, 7, 12])
    cnt = 0
    for i in range(field.size):
        sum = 0.0
        for j in range(field[i]):
            sum += prob[cnt + j]
        for j in range(field[i]):
            prob[cnt + j] /= sum
        cnt += field[i]

    '''
    # print the sampling probabilities
    cnt = 0
    for i in range(field_number):
        for j in range(field[i]):
            print(prob[cnt + j], end = ' ')
        cnt += field[i]
        print("")
    '''

    print("begin to calc all the train_X[i]")
    for instance in range(train_size):
        cnt = 0
        for i in range(field_number):
            p = np.zeros(field[i], dtype = float)
            for j in range(field[i]):
                p[j] = prob[cnt + j]
            sample = np.random.choice(field[i], 1, p = p)
            train_X[instance][cnt + sample[0]] = 1
            cnt += field[i]
        if (instance % 10000 == 0):
            print("calculated %d train_X[i]" % instance)

    print("begin to calc all the test_X[i]")
    for instance in range(test_size):
        cnt = 0
        for i in range(field_number):
            p = np.zeros(field[i], dtype = float)
            for j in range(field[i]):
                p[j] = prob[cnt + j]
            sample = np.random.choice(field[i], 1, p = p)
            test_X[instance][cnt + sample[0]] = 1
            cnt += field[i]
        if (instance % 10000 == 0):
            print("calculated %d test_X[i]" % instance)

#TODO: y[i] is too big now, you need to shrink it later

    print("begin to calc all the train_y[i]")
    for i in range(train_size):
        result = 0.0
        for j in range(dimension):
            result += train_X[i][j] * W[j]
        for j in range(dimension):
            for k in range(dimension):
                result += train_X[i][j] * train_X[i][k] * V[j][k]
        train_y[i] = float(result)
        if i % 10000 == 0:
            print("calculated %d train_y[i]" % i)

    print("begin to calc all the test_y[i]")
    for i in range(test_size):
        result = 0.0
        for j in range(dimension):
            result += train_X[i][j] * W[j]
        for j in range(dimension):
            for k in range(dimension):
                result += test_X[i][j] * test_X[i][k] * V[j][k]
        test_y[i] = float(result)
        # mx = max(mx, train_y[i])
        if i % 10000 == 0:
            print("calculated %d test_y[i]" % i)
    return train_X, train_y, test_X, test_y

def train(tr_X, tr_y, te_X, te_y):
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    train_X = np.asarray(tr_X).reshape(train_size, dimension)
    train_Y = np.asarray(tr_y).reshape(train_size, 1)

    X = tf.placeholder("float", [None, dimension])
    Y = tf.placeholder("float", [None, 1])
    print("X.shape = ", X.shape)
    print("Y.shape = ", Y.shape)

    W_o = tf.Variable(tf.random_uniform(shape = [dimension, 1], minval = -0.1, maxval = 0.1), name = "W_o")
    Bias_o = tf.Variable(tf.random_uniform(shape = [1], minval = -0.1, maxval = 0.1), name = "Bias_o")

    pred = tf.add(tf.multiply(X, W_o), Bias_o)
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, Y))))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                x = x.reshape(1, dimension)
                y = y.reshape(1, 1)
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                      "W_o=", sess.run(W_o), "Bias_o=", sess.run(Bias_o))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W_o=", sess.run(W_o), "Bias_o=", sess.run(Bias_o), '\n')

        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W_o) * train_X + sess.run(Bias_o), label='Fitted line')
        plt.legend()
        plt.show()

        # Testing example, as requested (Issue #2)
        test_X = np.asarray(te_X)
        test_y = np.asarray(te_y)

        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
            feed_dict={X: test_X, Y: test_y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        plt.plot(test_X, test_y, 'bo', label='Testing data')
        plt.plot(train_X, sess.run(W_o) * train_X + sess.run(Bias_o), label='Fitted line')
        plt.legend()
        plt.show()


make_parameters()
train_X, train_y, test_X, test_y = sampling()
train(tr_X = train_X, tr_y = train_y, te_X = test_X, te_y = test_y)
