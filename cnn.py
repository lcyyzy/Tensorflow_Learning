import tensorflow as tf
import pickle
import random
import os


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    with open('./data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    def separate(m):
        labels = []
        features = []
        for l, d in m:
            labels.append(l)
            features.append(d)
        return features, labels

    def sample_train(n=100):
        s = random.sample(train_data, n)
        return separate(s)

    sess = tf.InteractiveSession()

    width = 104
    height = 24
    x = tf.placeholder(tf.float32, [None, width, height])
    x_image = tf.reshape(x, [-1, width, height, 1])

    w_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    width //= 2
    height //= 2

    w_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    width //= 2
    height //= 2

    w_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    width //= 2
    height //= 2

    w_fc1 = weight_variable([width * height * 64, 512])
    b_fc1 = bias_variable([512])

    h_pool3_flat = tf.reshape(h_pool3, [-1, width * height * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([512, 30])
    b_fc2 = bias_variable([30])

    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 30])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    ckpt_file_dir = './cnn'
    ckpt_file_path = './cnn/cnn.ckpt'
    if not os.path.isdir(ckpt_file_dir):
        os.mkdir(ckpt_file_dir)

    try:
        saver.restore(sess, ckpt_file_path)
    except:
        sess.run(tf.global_variables_initializer())

    for i in range(5000):
        batch = sample_train()
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    saver.save(sess, ckpt_file_path)
    fs, ls = separate(test_data)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: fs, y_: ls, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
