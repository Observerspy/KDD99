import tensorflow as tf
import load_data as ld
import os
import numpy as np

class Config(object):
    n_features = 36
    n_height = 6
    n_classes = 3
    dropout = 0.5
    hidden_size_1 = 512
    hidden_size_2 = 256
    batch_size = 4096
    n_epochs = 100
    lr = 0.0001
    lamda = 0.01
    trainrate = 0.9



class NN(object):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, Config.n_height, Config.n_height, 1), name="input")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, Config.n_classes), name="label")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="drop")
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, labels_batch, dropout=1, is_training=False):
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch,
                 self.dropout_placeholder: dropout, self.is_training: is_training}
        return feed_dict

    def add_prediction_op(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('w_variable'):
            w1_init = xavier_initializer((3*3*32, Config.hidden_size_1))
            self.W1 = tf.Variable(w1_init, name="W1")
            tf.summary.histogram("W1", self.W1)
            w2_init = xavier_initializer((Config.hidden_size_1, Config.hidden_size_2))
            self.W2 = tf.Variable(w2_init, name="W2")
            tf.summary.histogram("W2", self.W2)
            w3_init = xavier_initializer((Config.hidden_size_2, Config.n_classes))
            self.W3 = tf.Variable(w3_init, name="W3")
            tf.summary.histogram("W3", self.W3)
        with tf.name_scope('b_variable'):
            b1 = tf.Variable(tf.zeros(Config.hidden_size_1), name="b1")
            tf.summary.histogram("b1", b1)
            b2 = tf.Variable(tf.zeros(Config.hidden_size_2), name="b2")
            tf.summary.histogram("b2", b2)
            b3 = tf.Variable(tf.zeros(Config.n_classes), name="b3")
            tf.summary.histogram("b3", b3)
        self.global_step = tf.Variable(0)

        with tf.name_scope('cnn'):
            layer_cnn = tf.layers.conv2d(self.input_placeholder, 32, 3, strides=1, padding='SAME')
            hidden = tf.nn.relu(layer_cnn)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [-1, shape[1] * shape[2] * shape[3]])
        with tf.name_scope('layer1'):
            layer1 = tf.matmul(reshape, self.W1) + b1
            h_batch1 = tf.layers.batch_normalization(layer1, center=True, scale=True, training=self.is_training)
            hidden1 = tf.nn.relu(h_batch1)
            tf.summary.histogram('hidden1_out', hidden1)
            h_drop1 = tf.nn.dropout(hidden1, self.dropout_placeholder)
        with tf.name_scope('layer2'):
            layer2 = tf.matmul(h_drop1, self.W2) + b2
            h_batch2 = tf.layers.batch_normalization(layer2, center=True, scale=True, training=self.is_training)
            hidden2 = tf.nn.relu(h_batch2)
            tf.summary.histogram('hidden2_out', hidden2)
            h_drop2 = tf.nn.dropout(hidden2, self.dropout_placeholder)
        with tf.name_scope('output'):
            pred = tf.matmul(h_drop2, self.W3) + b3
            tf.summary.histogram('out', pred)
        return pred

    def add_loss_op(self, pred):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        loss += Config.lamda * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) +
                        tf.nn.l2_loss(self.W3))
        tf.summary.scalar('loss', loss)
        return loss

    def add_training_op(self, loss):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            learning_rate = tf.train.exponential_decay(Config.lr, self.global_step, 1000, 0.8, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, merged, train_writer, i):
        feed = self.create_feed_dict(inputs_batch, labels_batch, self.config.dropout, True)
        rs, _, loss = sess.run([merged, self.train_op, self.loss], feed_dict=feed)
        train_writer.add_summary(rs, i)
        return loss

    def __init__(self, config):
        self.config = config
        self.build()

    def fit(self, sess, train_x, train_y):
        loss = self.train_on_batch(sess, train_x, train_y)

    def build(self):
        with tf.name_scope('inputs'):
            self.add_placeholders()
        with tf.name_scope('predict'):
            self.pred = self.add_prediction_op()
        with tf.name_scope('loss'):
            self.loss = self.add_loss_op(self.pred)
        with tf.name_scope('train'):
            self.train_op = self.add_training_op(self.loss)
        with tf.name_scope('accuracy'):
            self.acc_op = self.add_acc_op(self.pred,)
        self.argmax_op = self.add_argmax_ops(self.pred,)

    def add_acc_op(self, pred):
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(self.labels_placeholder, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def accuracy(self, sess, inputs_batch, labels_batch, name, merged, writer, i):
        feed = self.create_feed_dict(inputs_batch, labels_batch, 1, False)
        self.class_accuracy(sess, feed)
        rs, acc = sess.run([merged, self.acc_op], feed_dict=feed)
        writer.add_summary(rs, i)
        return acc

    def add_argmax_ops(self, pred):
        op = []
        pred_op = tf.argmax(pred, 1)
        op.append(pred_op)
        label_op = tf.argmax(self.labels_placeholder, 1)
        op.append(label_op)
        return op

    def class_accuracy(self, sess, feed):
        y_pre_argmax, v_y_argmax = sess.run(self.argmax_op, feed_dict=feed)
        for i in range(Config.n_classes):
            y_pre_mask = (y_pre_argmax != i)
            y_pre_class = np.ma.masked_array(y_pre_argmax, y_pre_mask)
            v_y_class = np.ma.masked_array(v_y_argmax, y_pre_mask)
            correct_prediction = np.equal(y_pre_class, v_y_class)
            acc = np.mean(correct_prediction)
            print("class:", i, "acc:", acc, end=" ")
        print("")

def main():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(PROJECT_ROOT, "data/train2.data")
    test_path = os.path.join(PROJECT_ROOT, "data/test2.data")
    train_log_path = os.path.join(PROJECT_ROOT, "log/train/")
    test_log_path = os.path.join(PROJECT_ROOT, "log/test/")
    dev_log_path = os.path.join(PROJECT_ROOT, "log/dev/")

    X, y = ld.load(train_path)#debug
    X = X.reshape([-1, 6, 6, 1])
    train_num = int(X.shape[0] * Config.trainrate)
    X_train = X[:train_num]
    y_train = y[:train_num]
    X_dev = X[train_num:-1]
    y_dev = y[train_num:-1]
    X_test, y_test = ld.load(test_path)
    X_test = X_test.reshape([-1, 6, 6, 1])
    print("train size :", X_train.shape, y_train.shape)
    print("dev size :", X_dev.shape, y_dev.shape)
    print("test size :", X_test.shape, y_test.shape)
    print("start training")

    with tf.Graph().as_default():
        config = Config()
        nn = NN(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)
        #必须在session外面
        y_train = tf.one_hot(y_train, depth=Config.n_classes)
        y_test = tf.one_hot(y_test, depth=Config.n_classes)
        y_dev = tf.one_hot(y_dev, depth=Config.n_classes)
        shuffle_batch_x, shuffle_batch_y = tf.train.shuffle_batch(
            [X_train, y_train], batch_size=Config.batch_size, capacity=10000,
            min_after_dequeue=5000, enqueue_many=True)

        with tf.Session() as session:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_log_path, session.graph)
            test_writer = tf.summary.FileWriter(test_log_path)
            dev_writer = tf.summary.FileWriter(dev_log_path)
            session.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            y_test, y_dev = session.run([y_test, y_dev])
            i = 0
            try:
                while not coord.should_stop():
                    #for i in range(Config.n_epochs * X_train.shape[0] // Config.batch_size):
                    #offset = (i * Config.batch_size) % (X_train.shape[0] - Config.batch_size)
                    #batch_x = X_train[offset:(offset + Config.batch_size), :]
                    #batch_y = y_train[offset:(offset + Config.batch_size)]
                    batch_x, batch_y = session.run([shuffle_batch_x, shuffle_batch_y])
                    loss = nn.train_on_batch(session, batch_x, batch_y, merged, train_writer, i)
                    i += 1
                    if i % 1000 == 0:
                        dev_acc = nn.accuracy(session, X_dev, y_dev, "dev", merged, dev_writer, i)
                        test_acc = nn.accuracy(session, X_test, y_test, "test", merged, test_writer, i)
                        print("step:", i, "loss:", loss, "dev_acc:", dev_acc, "test_acc:", test_acc)
                        saver.save(session, os.path.join(PROJECT_ROOT, "model/model_ckpt"), global_step=i)
            except tf.errors.OutOfRangeError:
                print("done")
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()