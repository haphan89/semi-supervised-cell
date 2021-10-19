import tensorflow as tf
import numpy as np
import os, io
from copy import deepcopy
from skimage.morphology import disk
from skimage.measure import label
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import center_of_mass
import time
import matplotlib.pyplot as plt

class M3(object):
    def __init__(self, session,
                 optimizer,
                 saver,
                 checkpoint_dir,
                 max_gradient=5,
                 summary_writer=None,
                 summary_every=100,
                 save_every=2000,
                 seq_len=10,
                 inference_z_jump=1,
                 batch_size=1,
                 training_data=None,
                 training_labels=None):

        self.run_context = True
        self.run_focus = True
        self.run_combined = True

        # Reduce resolution
        self.res_reduce = 3
        self.z_reduce = 1
        training_data = training_data[::self.z_reduce, ::self.res_reduce, ::self.res_reduce]
        training_data = np.expand_dims(training_data, -1)

        self.session = session
        self.optimizer = optimizer
        self.saver = saver
        self.max_gradient = max_gradient
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        # self.training = training

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.z_jump = inference_z_jump
        self.size_x = training_data.shape[1]
        self.size_y = training_data.shape[2]
        self.training_data = training_data[:110, :, :, :]  # np.vstack((training_data, training_data[-1:, :, :]))
        self.training_labels = training_labels
        self.test_data = training_data[110:, :, :, :]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.epoch_size = int(self.training_data.shape[0] / self.batch_size)

        self.create_variables()
        self.summary_writer.add_graph(self.session.graph)

        self.compress_jump = 1
        self.z_tolerance = 3
        self.z_tolerance_training = 1
        self.xy_tolerance = int(np.ceil(10/self.res_reduce))

    def create_variables(self):
        self.seq = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1],
                                  name='seq')
        self.event_location = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1],
                                             name='event_location')
        self.training = tf.placeholder(tf.bool, shape=())

        # MODEL OUTPUTS -------------------------------------------------------------------------------------
        self.focus_logits = self.model()
        if self.run_focus:
            self.focus_output = tf.sigmoid(self.focus_logits)
        else:
            self.focus_output = tf.zeros_like(self.seq)

        # self.focus_output *= self.matrix_contain_events
        # self.combined_logits = self.focus_logits * self.context_logits
        # ---------------------------------------------------------------------------------------------------

        # LOSSES --------------------------------------------------------------------------------------------
        self.focus_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.event_location, logits=self.focus_logits)) * 10

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'bias' not in v.name]) * 1e-5

        self.loss = self.focus_loss + self.lossL2  # + self.focus_loss + self.combined_loss
        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)
        # ---------------------------------------------------------------------------------------------------

        # SUMMARY -------------------------------------------------------------------------------------------
        ones = tf.ones_like(self.seq)

        focus_images = tf.concat([self.seq[0, :, :, :, :],
                                  ones[0, :, :, 0:2, :], self.focus_output[0, :, :, :, :],
                                  ones[0, :, :, 0:2, :], self.event_location[0, :, :, :, :] * self.seq[0, :, :, :, :],
                                  ones[0, :, :, 0:2, :], self.focus_output[0, :, :, :, :] * self.seq[0, :, :, :, :]], axis=-2)
        focus_sm = tf.summary.image("focus", focus_images, self.seq_len)

        cost_sm = tf.summary.scalar("cost", self.loss)
        focus_cost_sm = tf.summary.scalar("focus_cost", self.focus_loss)
        lossL2_cost_sm = tf.summary.scalar("lossL2_cost", self.lossL2)
        self.merge_list = [cost_sm, focus_cost_sm, lossL2_cost_sm,
                           focus_sm]
        self.summarize = tf.summary.merge(self.merge_list)

        # TP, FP, TN, FN -----------------------------------------------------------------------------------
        # th=1
        self.F1_1 = tf.placeholder_with_default(0.0, (), name='F1_1')
        self.precision_1 = tf.placeholder_with_default(0.0, (), name='precision_1')
        self.recall_1 = tf.placeholder_with_default(0.0, (), name='recal_1')

        # self.FN_events = tf.placeholder(tf.float32, [self.test_data.shape[0], self.size_x, self.size_y * 3, 1], name='FN_events')
        # FN_events_sm = tf.summary.image("FN_events", self.FN_events, self.test_data.shape[0])

        F1_sm = tf.summary.scalar("F1_1_cost", self.F1_1)
        precision_sm = tf.summary.scalar("precision_1_cost", self.precision_1)
        recall_sm = tf.summary.scalar("recall_1_cost", self.recall_1)

        self.test_merge_list = [F1_sm, precision_sm, recall_sm]
        self.test_summarize_1 = tf.summary.merge(self.test_merge_list)

        # th=3
        self.F1_3 = tf.placeholder_with_default(0.0, (), name='F1_3')
        self.precision_3 = tf.placeholder_with_default(0.0, (), name='precision_3')
        self.recall_3 = tf.placeholder_with_default(0.0, (), name='recall_3')

        # self.FN_events = tf.placeholder(tf.float32, [self.test_data.shape[0], self.size_x, self.size_y * 3, 1], name='FN_events')
        # FN_events_sm = tf.summary.image("FN_events", self.FN_events, self.test_data.shape[0])

        F1_sm = tf.summary.scalar("F1_3_cost", self.F1_3)
        precision_sm = tf.summary.scalar("precision_3_cost", self.precision_3)
        recall_sm = tf.summary.scalar("recall_3_cost", self.recall_3)

        self.test_merge_list = [F1_sm, precision_sm, recall_sm]
        self.test_summarize_3 = tf.summary.merge(self.test_merge_list)
        # ---------------------------------------------------------------------------------------------------

    def model(self, hidden_no=32):
        if self.run_focus:
            with tf.variable_scope('event_focus'):
                conv_lstm_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                           input_shape=[self.size_x, self.size_y, 1],
                                                           output_channels=hidden_no,
                                                           kernel_shape=[5, 5],
                                                           use_bias=True)
                conv_lstm_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                           input_shape=[self.size_x, self.size_y, 1],
                                                           output_channels=hidden_no,
                                                           kernel_shape=[5, 5],
                                                           use_bias=True)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(conv_lstm_fw, conv_lstm_bw,
                                                                  inputs=self.seq,
                                                                  dtype=tf.float32)

                # conv_lstm_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                #                                           input_shape=[self.size_x, self.size_y, 1],
                #                                           output_channels=hidden_no,
                #                                           kernel_shape=[5, 5],
                #                                           use_bias=True)
                # focus_convlstm_output, _ = tf.nn.dynamic_rnn(conv_lstm_2,
                #                                              inputs=tf.concat(outputs, -1),
                #                                              dtype=tf.float32)

                focus_convlstm_output = tf.layers.dropout(outputs, rate=0.3, training=self.training)
        else:
            focus_convlstm_output = tf.zeros_like(self.seq)

        with tf.variable_scope('outputs'):
            if self.run_focus:
                focus_convlstm_output = tf.squeeze(focus_convlstm_output)
                focus_convlstm_output = tf.reshape(focus_convlstm_output, [self.batch_size * self.seq_len, self.size_x, self.size_y, hidden_no])
                conv1 = tf.layers.conv2d(focus_convlstm_output, hidden_no, [5, 5], padding='same')
                conv2 = tf.layers.conv2d(conv1, 1, [1, 1], padding='same')
                focus_output = tf.reshape(conv2, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
            else:
                focus_output = tf.zeros_like(self.seq)

        return focus_output


    def get_sample(self):
        """
        self.training_data: [frame_id, x, y]
        self.training_labels: {frame_id, x, y} -> [n, 3]
        :return:
        """
        seq = np.zeros([self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        event_location = np.zeros([self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        for i in range(self.batch_size):
            z = np.random.randint(0, self.training_data.shape[0] - self.seq_len)
            seq_i = self.training_data[z:z + self.seq_len, :, :, :]
            labels_i = self.select_events(z=z)

            # Create matrices
            matrix_pos_i, matrix_neg_i, event_location_i = self.event_matrix(seq=seq_i, labels=labels_i)

            # Data augmentation
            flip_rotate_type = np.random.randint(0, 3)
            seq_i = self.flip_rotate(seq_i, type=flip_rotate_type)
            event_location_i = self.flip_rotate(event_location_i, type=flip_rotate_type)

            seq[i, :, :, :, :] = seq_i
            event_location[i, :, :, :, :] = event_location_i
        return seq, event_location

    def flip_rotate(self, seq, type=0):
        seq = np.squeeze(seq, -1)
        seq_2 = np.zeros(seq.shape)

        for i in range(seq.shape[0]):
            if type == 0:
                seq_2[i, :, :] = np.fliplr(seq[i, :, :])
            elif type == 1:
                seq_2[i, :, :] = np.flipud(seq[i, :, :])
            elif type == 2:
                seq_2[i, :, :] = np.rot90(seq[i, :, :], 2)
        seq_2 = np.expand_dims(seq_2, -1)
        return seq_2

    def select_events(self, z=0):
        z_min = z + 1
        z_max = z + self.seq_len - 1
        labels = self.training_labels[(self.training_labels[:, 0] >= z_min) & (self.training_labels[:, 0] <= z_max)]
        for i in range(labels.shape[0]):
            labels[i, 0] = labels[i, 0] - z
            labels[i, 1] = int(np.round(labels[i, 1] / self.res_reduce))
            labels[i, 2] = int(np.round(labels[i, 2] / self.res_reduce))
        return labels

    def event_matrix(self, seq=None, labels=None, z_range=8, xy_range=20, variance=8):
        matrix_pos = np.zeros_like(seq)
        event_location = np.zeros([seq.shape[0], seq.shape[1] + 50, seq.shape[2] + 50, 1])
        for i in range(labels.shape[0]):
            label = labels[i, :]

            z_0 = np.random.randint(label[0] - z_range, label[0] - 3)
            z_0 = 1 if z_0 < 1 else z_0
            z_1 = np.random.randint(label[0] + 3, label[0] + z_range)
            z_1 = self.seq_len - 1 if z_1 > self.seq_len - 1 else z_1

            x_0 = np.random.randint(label[1] - xy_range/2 - variance/2, label[1] - variance)
            x_0 = 0 if x_0 < 1 else x_0
            x_1 = x_0 + np.random.randint(xy_range, xy_range + variance)
            x_1 = seq.shape[1] if x_1 > seq.shape[1] else x_1

            y_0 = np.random.randint(label[2] - xy_range/2 - variance/2, label[2] - variance)
            y_0 = 0 if y_0 < 1 else y_0
            y_1 = y_0 + np.random.randint(xy_range, xy_range + variance)
            y_1 = seq.shape[2] if y_1 > seq.shape[2] else y_1

            matrix_pos[z_0:z_1, x_0:x_1, y_0:y_1, 0] = 1.0

            e_z = int(label[0])
            e_x = int(label[1]) + 25
            e_y = int(label[2]) + 25
            # print e_z, e_x, e_y, 'e_z, e_x, e_y'
            event_location[e_z - self.z_tolerance_training:e_z + self.z_tolerance_training + 1,
                            e_x - self.xy_tolerance + 1: e_x + self.xy_tolerance,
                            e_y - self.xy_tolerance + 1: e_y + self.xy_tolerance, :] = 1.0

        matrix_neg = np.ones_like(seq)
        matrix_neg[matrix_pos > 0] = 0.0

        event_location = event_location[:, 25:-25, 25:-25]
        # event_location = event_location * matrix_pos
        return matrix_pos, matrix_neg, event_location

    def non_event_matrix(self, seq=None, event_matrix_pos=None, z_range=8, xy_range=20, variance=8):
        non_event_matrix_neg = np.ones_like(seq)
        count = 0
        while count < 15:
            z_0 = np.random.randint(1, self.seq_len - 4)
            z_1 = z_0 + np.random.randint(6, z_range * 2)
            z_1 = self.seq_len - 1 if z_1 > self.seq_len - 1 else z_1
            x_0 = np.random.randint(0, event_matrix_pos.shape[1] - xy_range - variance)
            x_1 = x_0 + np.random.randint(xy_range, xy_range + variance)
            y_0 = np.random.randint(0, event_matrix_pos.shape[2] - xy_range - variance)
            y_1 = y_0 + np.random.randint(xy_range, xy_range + variance)

            if np.sum(event_matrix_pos[z_0:z_1, x_0:x_1, y_0:y_1]) == 0:
                non_event_matrix_neg[z_0:z_1, x_0:x_1, y_0:y_1, 0] = 0.0
                count += 1

        return non_event_matrix_neg


    def run_train(self):
        with self.session.as_default(), self.session.graph.as_default():
            print 'started ---', self.epoch_size, 'epoch_size'
            self.gs = self.session.run(self.global_step)
            try:
                while self.gs < 21000:
                    self.train_step()

                tf.logging.info("Reached global step {}. Stopping.".format(self.gs))
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            except KeyboardInterrupt:
                print 'a du ----'
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            return

    def train_step(self):
        start_time = time.time()
        # Get sample
        seq, event_location = self.get_sample()

        # emit summaries
        if self.gs % 30 != 1:
            feed_dict = {
                self.training: True,
                self.seq: seq,
                self.event_location: event_location
            }
            _, self.gs = self.session.run([self.train_op, self.global_step],
                                                         feed_dict)
            duration = time.time() - start_time
            if self.gs % 10 == 1:
                print duration, self.gs, 'duration, gs'

        else:
            feed_dict = {
                self.training: True,
                self.seq: seq,
                self.event_location: event_location
            }
            loss, summary, _, self.gs = self.session.run([self.loss, self.summarize, self.train_op, self.global_step],
                                                         feed_dict)
            duration = time.time() - start_time
            self.summary_writer.add_summary(summary, self.gs)
            print duration, self.gs, 'duration, gs, --- summary'

        if self.gs % 2000 == 100:
            print("Saving model checkpoint: {}".format(str(self.gs)))
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)

        if self.gs % 100 == 10:
            print 'testing ---'
            self.test()

    def get_test_sample(self, z=0):
        z0 = z
        z1 = z + self.seq_len
        seq = self.test_data_running[z0:z1, :, :, :]
        if z1 >= self.test_data.shape[0]:
            z_next = None
        else:
            z_next = z + self.z_jump
        seq = np.expand_dims(seq, 0)
        return seq, z_next

    def get_test_labels(self):
        labels = self.training_labels[self.training_labels[:, 0] >= 110]
        for i in range(labels.shape[0]):
            labels[i, 0] = labels[i, 0] - 110
            labels[i, 1] = int(np.round(labels[i, 1] / self.res_reduce))
            labels[i, 2] = int(np.round(labels[i, 2] / self.res_reduce))
        return labels

    def get_center(self, tensor):
        print np.sum(tensor), 'np.sum(tensor)---'
        if np.sum(tensor) > 0:
            # tensor = binary_dilation(tensor, iterations=1)

            labels = label(tensor)
            no_labels = len(np.unique(labels))
            print no_labels, 'no_labels'
            center = center_of_mass(tensor, labels, range(1, no_labels))
            # print center, 'center ---'
        else:
            center = []
        return center

    def create_gt_matrix(self):
        gt_matrix = np.zeros([self.test_data.shape[0] + 50, self.test_data.shape[1] + 50, self.test_data.shape[2] + 50])
        gt_matrix -= 1
        for i in range(self.test_labels.shape[0]):
            l = self.test_labels[i, :]
            l = [int(l[j]) for j in range(len(l))]
            z = l[0] + 25
            x = l[1] + 25
            y = l[2] + 25
            gt_matrix[z - self.z_tolerance:z + self.z_tolerance + 1, x - self.xy_tolerance: x + self.xy_tolerance + 1, \
                y - self.xy_tolerance: y + self.xy_tolerance + 1] = float(i)
        gt_matrix = gt_matrix[25:-25, 25:-25, 25:-25]
        return gt_matrix

    def create_gt_matrix_2(self):
        gt_matrix = np.zeros([self.test_data.shape[0] + 50, self.test_data.shape[1] + 50, self.test_data.shape[2] + 50])
        gt_matrix -= 1
        for i in range(self.test_labels.shape[0]):
            l = self.test_labels[i, :]
            l = [int(l[j]) for j in range(len(l))]
            z = l[0] + 25
            x = l[1] + 25
            y = l[2] + 25
            gt_matrix[z - 1:z + 2, x - self.xy_tolerance: x + self.xy_tolerance + 1, \
                y - self.xy_tolerance: y + self.xy_tolerance + 1] = float(i)
        gt_matrix = gt_matrix[25:-25, 25:-25, 25:-25]
        return gt_matrix

    def test(self, threshold=0.75, summarize=True):
        ori_test_data_shape = self.test_data.shape
        self.test_data_running = np.vstack((self.test_data, np.zeros_like(self.test_data)))
        self.test_labels = self.get_test_labels()

        z = 0; seq = True

        total_output = np.zeros_like(self.test_data)

        while z is not None:
            z_old = deepcopy(z)
            # print z, 'z'
            start_time = time.time()
            seq, z = self.get_test_sample(z)
            feed_dict = {
                self.training: False,
                self.seq: seq,
                self.event_location: np.zeros_like(seq)
            }
            output, _ = self.session.run([self.focus_output, self.global_step], feed_dict)
            duration = time.time() - start_time
            # print duration, 'duration'

            output[output >= threshold] = 1.0
            output[output < threshold] = 0.0
            total_output[z_old:z_old + self.seq_len, :, :, :] += output[0, :, :, :, :]

        total_output[total_output > 0] = 1.0

        total_output = total_output[:ori_test_data_shape[0], :, :, 0]
        center_pred = self.get_center(total_output)

        print center_pred[:5], 'center pred ===='

        for i in [1, 3]:
            TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
            self.z_tolerance = i
            self.gt_matrix = self.create_gt_matrix()
            correct_event_ids = []
            for i in range(len(center_pred)):
                c = center_pred[i]
                # print a[0], a[1], a[2], len(a), 'c---'
                if np.isnan(c[0]):
                    continue
                c = [int(np.round(c[j])) for j in range(len(c))]
                event_id = self.gt_matrix[c[0], c[1], c[2]]
                event_id = int(event_id)
                if event_id < 0:
                    FP += 1.0
                else:
                    xy_dist = np.sqrt((c[1] - self.test_labels[event_id, 1]) ** 2 + (c[2] - self.test_labels[event_id, 2]) ** 2)
                    if xy_dist <= self.xy_tolerance:
                        if event_id not in correct_event_ids:
                            TP += 1.0
                            correct_event_ids.append(event_id)
                    else:
                        FP += 1.0

            FN = self.test_labels.shape[0] - TP
            print TP, FP, TN, FN, 'TP, FP, TN, FN', len(self.test_labels), 'self.test_labels'

            precision = (TP / (TP + FP)) * 100 if (TP + FP) != 0 else 0
            recall = (TP / (TP + FN)) * 100 if (TP + FN) != 0 else 0
            F1_score = 2 * (precision * recall / (precision + recall)) if (precision + recall) != 0 else 0.0
            print precision, recall, F1_score, 'precision, recall, F1_score'
            print self.xy_tolerance, 'self.xy_tolerance'
            print self.z_tolerance, 'self.z_tolerance'

            # SUMMARY
            if summarize:
                if self.z_tolerance == 1:
                    feed_dict = {
                        self.F1_1: F1_score,
                        self.precision_1: precision,
                        self.recall_1: recall,
                    }
                    summary, _ = self.session.run([self.test_summarize_1, self.global_step], feed_dict)
                    self.summary_writer.add_summary(summary, self.gs)
                else:
                    feed_dict = {
                        self.F1_3: F1_score,
                        self.precision_3: precision,
                        self.recall_3: recall,
                    }
                    summary, _ = self.session.run([self.test_summarize_3, self.global_step], feed_dict)
                    self.summary_writer.add_summary(summary, self.gs)
