import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import os

# git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg
from tensorflow_vgg import vgg16
from tensorflow_vgg import vgg_weight_utils
from kaggle_amazon_data import KaggleAmazonData
from arg_utils import process_args

import sklearn.metrics as metrics


class KaggleAmazon(object):
    def __init__(self, train_vgg=False, fc1_shape=1024, fc2_shape=256, pos_weight=1.0, run_id=None):
        self._train_vgg = train_vgg
        self._fc1_shape = fc1_shape
        self._fc2_shape = fc2_shape
        self._pos_weight = pos_weight
        self._run_id = run_id

        self._data_root = os.path.join(os.path.expanduser("~"), 'Developer/data/kaggle_amazon/')
        self.data_source = KaggleAmazonData(
            training_data_dir=os.path.join(self._data_root, 'train-tif-v2'),
            training_data_dir_jpeg=os.path.join(self._data_root, 'train-jpg'),
            training_labels_file=os.path.join(self._data_root, 'train_v2.csv'),
            testing_data_dir=os.path.join(self._data_root, 'test-tif-v2'),
            validation_percentage=1.0)

        self._labels_stats = self.data_source.labels_stats
        self._classifier_count = len(self._labels_stats.labels_set)

        self.setup_classifiers()

    def single_classifier(self, input, c_index):
        """
        Build a single classifier (on top of input) for the c_index^th category
        """
        label = '%d_%s' % (c_index, self._labels_stats.labels_dict_inv[c_index])
        print('Building classifier for', label)
        with tf.variable_scope(label, reuse=False):
            fc1 = tf.nn.dropout(tf.layers.dense(input, self._fc1_shape, activation=None, name='fc1'), self.dropout_ratio_)
            if self._fc2_shape == 0:
                fc2 = fc1
            else:
                fc2 = tf.nn.dropout(tf.layers.dense(fc1, self._fc2_shape, activation=None, name='fc2'), self.dropout_ratio_)
            logits = tf.layers.dense(fc2, 1, activation=None, name='logits')

            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
            #                                                               labels=self.labels_[:, c_index:c_index + 1]))
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits,
                                                                           targets=self.labels_[:, c_index:c_index + 1],
                                                                           pos_weight=self._pos_weight))

            predictions = tf.sigmoid(logits)

            _predictions = tf.cast(tf.transpose(tf.round(predictions)), tf.int8)
            _labels = tf.cast(self.labels_[:, c_index], tf.int8)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(_predictions, _labels), dtype=tf.float32))

            return logits, predictions, loss, accuracy

    def setup_classifiers(self):
        """
        Setup the whole network
        """
        self.inputs_ = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
        self.labels_ = tf.placeholder(tf.float32, shape=[None, self._classifier_count], name='labels')
        self.learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_ratio_ = tf.placeholder(tf.float32, name='dropout_ratio')
        self.smooth_ = tf.placeholder(tf.float32, name='smooth')

        # load VGG
        vgg_weight_utils.download_vgg_parameter_file()
        self.vgg = vgg16.Vgg16(vgg16_npy_path='./data/vgg16.npy', trainable=self._train_vgg)  # initiates weights from vgg16.npy by default
        with tf.variable_scope('vgg'):
            self.vgg.build(self.inputs_)

        # set up a classifier for each category
        self.logits = []
        self.predictions = []
        self.accuracies = []
        self.losses = []
        for c_index in range(self._classifier_count):
            c_logits, c_prediction, c_loss, c_accuracy = self.single_classifier(self.vgg.relu6, c_index)

            self.logits.append(c_logits)
            self.predictions.append(c_prediction)
            self.losses.append(c_loss)
            self.accuracies.append(c_accuracy)

        self.loss = tf.add_n(self.losses)
        self.accuracy = tf.add_n(self.accuracies) / float(self._classifier_count)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.loss)

    def train(self, batch_size=60, epochs=50, learning_rate=0.0001, dropout_ratio=1.0, smooth=1.0):
        _training_cost_summary = tf.summary.scalar('training_loss', self.loss)
        _training_accuracy_summary = tf.summary.scalar('training_accuracy', self.accuracy)
        merged_summaries = tf.summary.merge_all()

        run_id = 'b_%d_e_%d_lr_%.6f_drp_%.2f_sm_%.2f_pw_%.2f_fc1_%d_fc2_%d' % \
                 (batch_size, epochs, learning_rate, dropout_ratio, smooth, self._pos_weight, self._fc1_shape, self._fc2_shape) + \
                 ('_vgg_on' if self._train_vgg else '_vgg_off') + ('_' + self._run_id if self._run_id is not None else '')
        print('Training model ' + run_id + '\n')
        self.checkpoint_name = 'ka_' + run_id + '.ckpt'

        summary_writer = tf.summary.FileWriter(os.path.join(self._data_root, 'logs/' + run_id))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            summary_writer.add_graph(sess.graph)
            summary_writer.flush()

            iteration = 0
            for e in range(epochs):
                for _ in range(self.data_source.training_size // batch_size):
                    _, rgb, _, labels = self.data_source.random_training_batch(batch_size=batch_size)

                    sess.run(self.optimizer,
                             feed_dict={self.inputs_: rgb, self.labels_: labels,
                                        self.learning_rate_: learning_rate,
                                        self.dropout_ratio_: dropout_ratio,
                                        self.smooth_: smooth})

                    if iteration % 5 == 0:
                        loss, accuracy, summary = sess.run([self.loss, self.accuracy, merged_summaries],
                                                           feed_dict={self.inputs_: rgb, self.labels_: labels,
                                                                      self.dropout_ratio_: 1.0})
                        summary_writer.add_summary(summary, iteration)

                        print("Epoch: {}/{}".format(e + 1, epochs),
                              "Iteration: {}".format(iteration),
                              "Training loss: {:.5f}".format(loss),
                              "Training accuracy: {:.5f}".format(accuracy))

                    if iteration % 100 == 0:
                        _training_accuracy, _training_f2_score = self._eval_labelled_images(
                            sess=sess, image_source=self.data_source.training_image, sample_count=500, verbose=False)
                        _validation_accuracy, _validation_f2_score = self._eval_labelled_images(
                            sess=sess, image_source=self.data_source.validation_image,
                            sample_count=self.data_source.validation_size, verbose=False)

                        summary_writer.add_summary(tf.Summary(
                            value=[tf.Summary.Value(tag='training_accuracy_debug', simple_value=_training_accuracy)]),
                            global_step=iteration)
                        summary_writer.add_summary(tf.Summary(
                            value=[tf.Summary.Value(tag='training_f2_score', simple_value=_training_f2_score)]),
                            global_step=iteration)
                        summary_writer.add_summary(
                            tf.Summary(value=[tf.Summary.Value(tag='validation_accuracy', simple_value=_validation_accuracy)]),
                            global_step=iteration)
                        summary_writer.add_summary(
                            tf.Summary(value=[tf.Summary.Value(tag='validation_f2_score', simple_value=_validation_f2_score)]),
                            global_step=iteration)

                        print("-"*len("Epoch: {}/{}".format(e + 1, epochs) + " Iteration: {}".format(iteration) + "Training loss: {:.5f} ".format(loss)),
                              "Training accuracy: {:.5f}, f2_score: {:.5f}".format(_training_accuracy, _training_f2_score))
                        print("-"*len("Epoch: {}/{}".format(e + 1, epochs) + " Iteration: {}".format(iteration) + "Training loss: {:.5f} ".format(loss)),
                              "Vldation accuracy: {:.5f}, f2_score: {:.5f}".format(_validation_accuracy, _validation_f2_score))

                        _peek = self._peek(sess, self.vgg.conv5_3)
                        print(run_id, '\npeeeekabooo: ', _peek.sum())

                    iteration += 1

                saver.save(sess, os.path.join(self._data_root, 'checkpoints', self.checkpoint_name))

    def _peek(self, sess, tensor):
        x = np.zeros([1, 224, 224, 3])
        tensor_value = sess.run(tensor, feed_dict={self.inputs_: x, self.dropout_ratio_: 1.0})
        return tensor_value

    def eval_set(self, image_source, saved_session_id, sample_count=10, plot=False):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # p1 = self._peek(sess, self.vgg.conv5_3)
            saver.restore(sess, os.path.join(self._data_root, 'checkpoints', saved_session_id))
            # p2 = self._peek(sess, self.vgg.conv5_3)
            # print(p1.sum(), ' vs ', p2.sum())

            with open(self._run_id+'.csv', 'w') as result:
                result.write('image_name,tags'+'\n')
                print('image_name,tags')

            for i in range(sample_count):
                image_name, rgb, ir, true_labels = image_source(i)

                predictions = sess.run([self.predictions[i] for i in range(self._classifier_count)],
                                       feed_dict={self.inputs_: rgb, self.dropout_ratio_: 1.0})
                predictions = np.round(np.array(predictions).squeeze()).astype(np.int)
                predicted_labels = np.where(predictions == 1)[0]
                predicted_labels_str = image_name + ',' + ' '.join([self._labels_stats.labels_dict_inv[label] for label in predicted_labels])

                with open(self._run_id + '.csv', 'a') as result:
                    result.write(predicted_labels_str+'\n')
                # print(predicted_labels_str)

                true_labels_str = None
                if true_labels is not None:
                    true_labels = true_labels.squeeze().astype(np.int)
                    accuracy = np.mean(np.equal(predictions, true_labels).astype(np.float))
                    true_labels_str = ', '.join([self._labels_stats.labels_dict_inv[label] for label in
                                                 np.where(true_labels == 1)[0]])

                    print('vs')
                    print(true_labels_str)
                    print(accuracy)

                    true_labels_str += '  % = {:.4f}'.format(accuracy)

                if plot:
                    self.show_image_and_labels(
                        rgb=rgb.squeeze(),
                        predicted_labels=', '.join([self._labels_stats.labels_dict_inv[label] for label in predicted_labels]),
                        true_labels=true_labels_str)

    def show_image_and_labels(self, rgb, predicted_labels, true_labels=None):
        plt.figure(1)
        plt.ion()
        plt.imshow(rgb)
        label = predicted_labels
        if true_labels is not None:
            label += ' vs. ' + true_labels
        plt.title(label)
        plt.draw()
        plt.waitforbuttonpress()

    def _eval_labelled_images(self, sess, image_source, sample_count=200, verbose=False):
        accuracies = []
        f2_scores = []
        max_batch_size = 100  # experimental number; fits in memory
        for i in range(0, sample_count, max_batch_size):
            image_names, rgb, ir, labels = image_source(range(i, min(sample_count, i+max_batch_size)))

            # logits = sess.run([self.logits[i] for i in range(self.classifier_count)],
            #                   feed_dict={self.inputs_: rgb})
            # print([logit[0][0] for logit in logits])

            predictions = sess.run([self.predictions[i] for i in range(self._classifier_count)],
                                   feed_dict={self.inputs_: rgb, self.dropout_ratio_: 1.0})

            _labels = labels.astype(np.int)
            _predictions = np.round(np.array(predictions)).squeeze().transpose().astype(np.int)
            _accuracy = np.mean(np.equal(_labels, _predictions).astype(np.float))
            _f2_score = metrics.fbeta_score(_labels, _predictions, beta=2, average="samples")

            accuracies.append(_accuracy)
            f2_scores.append(_f2_score)

            if verbose:
                print(_labels)
                print(_predictions)
                print(_labels - _predictions)
                print(np.equal(_labels, _predictions))
                print(_accuracy)
                print("")

        avrg_accuracy = np.mean(np.array(accuracies))
        avrg_f2_score = np.mean(np.array(f2_scores))
        if verbose:
            print("Average accuracy: ", avrg_accuracy)
            print("Average f2_score: ", avrg_f2_score)

        return avrg_accuracy, avrg_f2_score

    def eval_labelled_set(self, image_source, saved_session_id, sample_count=10, verbose=True):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(self._data_root, 'checkpoints', saved_session_id))
            self._eval_labelled_images(sess, image_source, sample_count=sample_count, verbose=verbose)


if __name__ == '__main__':
    train_vgg, learning_rate, dropout_rate, batch_size, epochs, smooth, pos_weight, fc1, fc2 = process_args()

    trn = KaggleAmazon(train_vgg=train_vgg, pos_weight=pos_weight, fc1_shape=fc1, fc2_shape=fc2,
                       run_id="{:%m_%d_%H_%M}".format(datetime.datetime.now()))
    trn.train(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
              dropout_ratio=dropout_rate, smooth=smooth)
    trn.eval_set(image_source=trn.data_source.testing_image,
                 saved_session_id=trn.checkpoint_name,
                 sample_count=len(trn.data_source.testing_files))


    # trn.eval_labelled_set(image_source=trn.data_source.validation_image,
    #                       saved_session_id='ka_degub.ckpt', sample_count=10)
    #
    # trn.eval_set(image_source=trn.data_source.validation_image,
    #              saved_session_id='ka_degub.ckpt', sample_count=10)

    # trn.eval_set(image_source=trn.data_source.training_image,
    #              saved_session_id='ka_b_10_e_1_lr_0.00100_vgg_on_debug.ckpt')

    # trn.eval_set(image_source=trn.data_source.testing_image,
    #              saved_session_id='ka_b_60_e_100_lr_0.00010_drp_0.50_vgg_on_bruhaha.ckpt',
    #              sample_count=len(trn.data_source.testing_files))

