import tensorflow as tf
import os
import time
import numpy as np
from copy import deepcopy
from scipy.ndimage import label

class M3(object):
    def __init__(self, session,
                 optimizer,
                 saver,
                 checkpoint_dir,
                 summary_writer=None,
                 summary_every=100,
                 save_every=2000,
                 training=True,
                 seq_len=20,
                 no_cuts=1,
                 batch_size=1,
                 training_data=None,
                 training_frames=None):
        self.session = session
        self.optimizer = optimizer
        self.saver = saver
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.training = training

        self.segment_groups = 6

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.no_cuts = no_cuts
        self.size_x = training_data.shape[1]
        self.size_y = training_data.shape[2]
        self.training_data = training_data
        self.training_frames = training_frames  # frames: [frame_id, x, y, 1]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.create_variables()
        self.summary_writer.add_graph(self.session.graph)

        self.compress_jump = 1
        self.z_tolerance = 1
        self.xy_tolerance = 10

    def create_variables(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input_x')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input_y')
        self.frames = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='frames')

        self.predict_output = self.predict(self.x)

        self.loss_predict = tf.losses.mean_squared_error(labels=self.y, predictions=self.predict_output) * 10

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'bias' not in v.name]) * 1e-5

        # self.loss = self.loss_dropout + self.loss_reconstr  # + self.loss_probs_diversity
        self.loss = self.loss_predict + self.lossL2

        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)

        y_mask = tf.where(tf.greater(self.predict_output, 0.5),
                          tf.constant(1.0, shape=self.predict_output.shape, dtype=tf.float32),
                          tf.constant(0.0, shape=self.predict_output.shape, dtype=tf.float32))
        self.predict_output = y_mask

        ones = tf.ones_like(self.x[0, :, :, :, :])
        predict_images = tf.concat((self.x[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.predict_output[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.y[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.frames[0, :, :, :, :] / tf.reduce_max(self.frames[0, :, :, :, :])), 2)
        """
        x_mask = tf.where(tf.greater(self.x, 1),
                                 tf.constant(1.0, shape=self.x.shape, dtype=tf.float32),
                                 tf.constant(0.0, shape=self.x.shape, dtype=tf.float32))
        y_images = tf.concat((self.x[0, :, :, :, :] * x_mask[0, :, :, :, :],
                                ones[:, :, 0:1, :],
                              self.y[0, :, :, :, :]), 2)
        """

        predict_images_sm = tf.summary.image("predict_images", predict_images, self.seq_len)
        # y_images_sm = tf.summary.image("y_images", y_images, self.seq_len)

        cost_sm = tf.summary.scalar("cost", self.loss)
        cost_predict_sm = tf.summary.scalar("cost_predict", self.loss_predict)
        cost_l2_sm = tf.summary.scalar("cost_l2", self.lossL2)
        self.merge_list = [predict_images_sm,
                           cost_sm, cost_predict_sm,
                           cost_l2_sm]
        self.summarize = tf.summary.merge(self.merge_list)

        self.summarize_2 = tf.summary.merge([predict_images_sm])
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

    def preprocess(self, input):
        input = tf.reshape(input, [self.batch_size * self.seq_len, self.size_x, self.size_y, 1])

        kernel_tensor = tf.constant(0.5, shape=[3, 3, 1])
        input = tf.nn.erosion2d(input, kernel=kernel_tensor, strides=[1, 1, 1, 1],
                                        rates=[1, 1, 1, 1], padding='SAME')
        input_mean = tf.reduce_mean(input)
        input = tf.where(tf.greater(input, input_mean),
                                 tf.constant(1.0, shape=input.shape, dtype=tf.float32),
                                 tf.constant(0.0, shape=input.shape, dtype=tf.float32))
        input = tf.reshape(input, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        return input

    def predict(self, x, hidden_no=16):
        x = tf.reshape(x, [self.batch_size * self.seq_len, self.size_x, self.size_y, 1])
        conv1 = tf.layers.conv2d(x, hidden_no, [5, 5], padding='same')
        conv2 = tf.layers.conv2d(conv1, hidden_no, [5, 5], padding='same')
        conv2 = tf.reshape(conv2, [self.batch_size, self.seq_len, self.size_x, self.size_y, hidden_no])

        conv_lstm_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                  input_shape=[self.size_x, self.size_y, hidden_no],
                                                  output_channels=hidden_no,
                                                  kernel_shape=[5, 5],
                                                  use_bias=True)
        convlstm_output_fw, _ = tf.nn.dynamic_rnn(conv_lstm_fw, inputs=conv2, dtype=tf.float32, scope='convlstm1')

        """
        conv_lstm_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, hidden_no],
                                                   output_channels=hidden_no,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        convlstm_output_bw, _ = tf.nn.dynamic_rnn(conv_lstm_bw, inputs=conv2[:, ::-1, :, :, :], dtype=tf.float32, scope='convlstm2')

        convlstm_output = tf.concat((convlstm_output_fw, convlstm_output_bw), axis=-1)
        """
        convlstm_output = tf.reshape(convlstm_output_fw, [self.batch_size * self.seq_len, self.size_x, self.size_y, hidden_no])

        conv3 = tf.layers.conv2d(convlstm_output, hidden_no, [5, 5], padding='same')
        conv3 = tf.layers.conv2d(conv3, 1, [1, 1], padding='same')
        predict = tf.reshape(conv3, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        return predict

    def get_sample(self):
        n = np.random.randint(0, self.training_data.shape[0] - self.seq_len, size=self.batch_size)
        segment_seq = np.transpose(np.array([self.training_data[n + i, :, :, :] for i in range(self.seq_len)]),
                              [1, 0, 2, 3, 4])
        segment_seq[segment_seq > 0] = 1
        x = np.zeros_like(segment_seq)
        y = np.zeros_like(segment_seq)

        for i in range(self.batch_size):
            labels, _ = label(segment_seq[i, :, :, :])
            x1, y1 = self.remove_cells(labels)
            x[i, :, :, :] += x1
            y[i, :, :, :] += y1
        frames = np.transpose(np.array([self.training_frames[n + i, :, :, :] for i in range(self.seq_len)]),
                                [1, 0, 2, 3, 4])
        return x, y, frames

    def remove_cells(self, segmentation, count=16):
        segment_ids = np.unique(segmentation)
        select = np.arange(segment_ids.shape[0])
        np.random.shuffle(select)
        segment_ids = [segment_ids[i] for i in select[:count] if segment_ids[i] > 0]

        x = np.zeros_like(segmentation)
        x[segmentation > 0] = 1
        y = np.zeros_like(segmentation)

        for id in segment_ids:
            y_0 = np.zeros_like(segmentation)
            y_0[segmentation == id] = 1
            y_sum = np.sum(y_0, axis=(1, 2))
            inds = np.where(y_sum > 0)[0]
            if len(inds) >= 2:
                begin, end = inds[0], inds[-1]
                y[begin + 1:end + 1, :, :] += y_0[begin + 1:end + 1, :, :]
                y_0[begin, :, :] = y_0[begin, :, :] * -1
                # y_0[end, :, :] = y_0[end, :, :] * -2
                x -= y_0
        # x = x - y
        return x, y

    def get_inference_frames(self, x=0):
        more_frames = True
        # n = np.random.randint(0, self.training_data.shape[0] - self.seq_len, size=self.batch_size)
        n = np.arange(x, np.minimum(x + self.batch_size, self.training_data.shape[0] - self.seq_len))
        segment_seq = np.transpose(np.array([self.training_data[n + i, :, :, :] for i in range(self.seq_len)]),
                                   [1, 0, 2, 3, 4])

        frames = np.transpose(np.array([self.training_frames[n + i, :, :, :] for i in range(self.seq_len)]),
                              [1, 0, 2, 3, 4])

        shape = segment_seq.shape
        zeros_size = 0
        if shape[0] < self.batch_size:
            segment_seq = np.concatenate((segment_seq, np.zeros([self.batch_size - segment_seq.shape[0]] + list(segment_seq.shape[1:]))), 0)
            frames = np.concatenate((frames, np.zeros([self.batch_size - frames.shape[0]] + list(frames.shape[1:]))), 0)
            more_frames = False
            zeros_size = self.batch_size - shape[0]
        return segment_seq, frames, x + self.batch_size, more_frames, zeros_size

    def run_train(self):
        with self.session.as_default(), self.session.graph.as_default():
            print 'started ---'
            self.gs = self.session.run(self.global_step)
            try:
                while self.gs <= 5100:
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
        x, y, frames = self.get_sample()

        feed_dict = {
            self.x: x,
            self.y: y,
            self.frames: frames
        }

        loss, summary, _, self.gs = self.session.run([
            self.loss, self.summarize, self.train_op, self.global_step], feed_dict)
        duration = time.time() - start_time

        # emit summaries
        if self.gs % 10 == 1:
            print duration, self.gs, 'duration, self.gs'
            print loss, 'loss'

        if self.gs % 10 == 4:
            print 'summary ---'
            self.summary_writer.add_summary(summary, self.gs)

        if self.gs % 2000 == 100:
            print("Saving model checkpoint: {}".format(str(self.gs)))
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)

    def inference(self, ends_segments):
        ends_segments[ends_segments >= 3.0] = 1.0
        self.training_data[self.training_data > 0] = 1
        self.training_data = self.training_data + ends_segments

        more_frames = True
        x = 0
        big_probs = None
        zeros_size = 0
        while more_frames:
            print x, 'x'
            segment_seq, frames, x, more_frames, zeros_size = self.get_inference_frames(x)

            start_time = time.time()
            feed_dict = {
                self.x: segment_seq,
                self.y: np.zeros_like(segment_seq),
                self.frames: frames
            }

            probs, summary, _, self.gs = self.session.run([
                self.predict_output, self.summarize_2, self.increment_global_step_op, self.global_step], feed_dict)
            duration = time.time() - start_time
            print duration, 'sec', self.gs
            self.summary_writer.add_summary(summary, self.gs)

            probs[probs <= 0.5] = 0
            probs[probs > 0.5] = 1
            big_probs = probs if big_probs is None else np.vstack((big_probs, probs))
        if zeros_size > 0:
            big_probs = big_probs[:-zeros_size, :, :, :]

        predicts = np.zeros_like(self.training_data, dtype=np.float32)
        for i in range(big_probs.shape[0]):
            a = big_probs[i, :, :, :, :]
            b = np.zeros_like(a, dtype=np.float32)
            b[a > 0.5] = 1.0
            predicts[i:i+self.seq_len, :, :, :] += b
        predicts[predicts > 0] = 1

        frames = np.repeat(self.training_frames, self.segment_groups, -1)
        segmented_frames = predicts * frames
        np.savez_compressed('/media/newhd/Ha/my_env/cell_5_segmentation_F0005/events_prediction.npz',
                            predicts=predicts, predicts_frames=segmented_frames)




