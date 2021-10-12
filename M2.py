import tensorflow as tf
import os
import time
import numpy as np
from copy import deepcopy
from scipy.ndimage import label, binary_dilation

class M2(object):
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
                 training_labels=None,
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
        self.size_x = training_labels.shape[1]
        self.size_y = training_labels.shape[2]
        self.training_labels = training_labels
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
        self.segmented_zones = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input_segmented_zones')
        self.frames = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='frames')

        self.segments = self.output_segments(self.x)

        self.loss_segments = tf.losses.mean_squared_error(labels=self.y * self.segmented_zones,
                                                         predictions=self.segments * self.segmented_zones)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'bias' not in v.name]) * 1e-5

        # self.loss = self.loss_dropout + self.loss_reconstr  # + self.loss_probs_diversity
        self.loss = self.loss_segments + self.lossL2

        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)

        ones = tf.ones_like(self.x[0, :, :, :, :])
        segments_images = tf.concat((self.x[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.segments[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.y[0, :, :, :, :],
                                    ones[:, :, 0:1, :],
                                    self.frames[0, :, :, :, :] / tf.reduce_max(self.frames[0, :, :, :, :])), 2)

        segments_images_sm = tf.summary.image("segments_images", segments_images, self.seq_len)
        # y_images_sm = tf.summary.image("y_images", y_images, self.seq_len)

        cost_sm = tf.summary.scalar("cost", self.loss)
        cost_segments_sm = tf.summary.scalar("cost_segments", self.loss_segments)
        cost_l2_sm = tf.summary.scalar("cost_l2", self.lossL2)
        self.merge_list = [segments_images_sm,
                           cost_sm, cost_segments_sm,
                           cost_l2_sm]
        self.summarize = tf.summary.merge(self.merge_list)

        self.summarize_2 = tf.summary.merge([segments_images_sm])
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

    def output_segments(self, x, hidden_no=32):
        x = tf.reshape(x, [self.batch_size * self.seq_len, self.size_x, self.size_y, 1])
        conv1 = tf.layers.conv2d(x, hidden_no, [5, 5], padding='same')
        conv2 = tf.layers.conv2d(conv1, hidden_no, [5, 5], padding='same')
        conv2 = tf.reshape(conv2, [self.batch_size, self.seq_len, self.size_x, self.size_y, hidden_no])

        conv_lstm = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                input_shape=[self.size_x, self.size_y, hidden_no],
                                                output_channels=hidden_no,
                                                kernel_shape=[5, 5],
                                                use_bias=True)
        convlstm_output, state = tf.nn.dynamic_rnn(conv_lstm, inputs=conv2, dtype=tf.float32, scope='convlstm1')

        convlstm_output = tf.reshape(convlstm_output,
                                     [self.batch_size * self.seq_len, self.size_x, self.size_y, hidden_no])

        conv3 = tf.layers.conv2d(convlstm_output, hidden_no, [5, 5], padding='same')
        conv3 = tf.layers.conv2d(conv3, 1, [1, 1], padding='same')
        segments = tf.reshape(conv3, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        return segments

    def get_sample(self):
        n = np.random.randint(0, self.training_labels.shape[0] - self.seq_len, size=self.batch_size)
        segment_seq = np.transpose(np.array([self.training_labels[n + i, :, :, :] for i in range(self.seq_len)]),
                              [1, 0, 2, 3, 4])
        frames = np.transpose(np.array([self.training_frames[n + i, :, :, :] for i in range(self.seq_len)]),
                              [1, 0, 2, 3, 4])
        x = deepcopy(frames)
        segment_seq[segment_seq > 0] = 1
        y = deepcopy(segment_seq)
        segmented_zones = self.create_segmented_zones(segment_seq)

        for i in range(self.batch_size):
            x1 = self.random_dropout_masks(x[i, :, :, :])
            x[i, :, :, :] += x1
        return x, y, segmented_zones, frames

    def create_segmented_zones(self, segments):
        masks = deepcopy(segments)
        for i in range(masks.shape[0]):
            masks[i, :, :, :] = binary_dilation(masks[i, :, :, :], iterations=3)
        masks[masks > 0] = 1.0
        return masks

    def random_dropout_masks(self, frames, holes=20):
        masks = np.ones_like(frames)
        masks_margin = np.zeros([frames.shape[0] + 100, frames.shape[1] + 200, frames.shape[2] + 200])
        masks_margin[:-100, 100:frames.shape[1] + 100, 100:frames.shape[2] + 100] = masks

        z_ids = np.random.randint(0, frames.shape[0], size=holes)
        x_ids = np.random.randint(0, frames.shape[1], size=holes) + 100
        y_ids = np.random.randint(0, frames.shape[2], size=holes) + 100
        hole_sizes = np.random.randint(8, 30, size=holes)
        hole_lengths = np.random.randint(1, 15, size=holes)

        for i in range(holes):
            z, x, y = z_ids[i], x_ids[i], y_ids[i]
            hole_size, hole_length = hole_sizes[i], hole_lengths[i]
            masks_margin[z:z+hole_length, x:x+hole_size, y:y+hole_size] = 0.0

        masks = masks_margin[:-100, 100:frames.shape[1] + 100, 100:frames.shape[2] + 100]
        return masks

    def get_inference_frames(self, x=0):
        more_frames = True
        # n = np.random.randint(0, self.training_labels.shape[0] - self.seq_len, size=self.batch_size)
        n = np.arange(x, np.minimum(x + self.batch_size, self.training_labels.shape[0] - self.seq_len))
        segment_seq = np.transpose(np.array([self.training_labels[n + i, :, :, :] for i in range(self.seq_len)]),
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
                while self.gs <= 2500:
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
        x, y, segmented_zones, frames = self.get_sample()

        feed_dict = {
            self.x: x,
            self.y: y,
            self.segmented_zones: segmented_zones,
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
        self.training_labels[self.training_labels > 0] = 1
        self.training_labels = self.training_labels + ends_segments

        more_frames = True
        x = 0
        big_segments = None
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

            segments, summary, _, self.gs = self.session.run([
                self.segments, self.summarize_2, self.increment_global_step_op, self.global_step], feed_dict)
            duration = time.time() - start_time
            print duration, 'sec', self.gs
            self.summary_writer.add_summary(summary, self.gs)

            segments[segments <= 0.5] = 0
            segments[segments > 0.5] = 1
            big_segments = segments if big_segments is None else np.vstack((big_segments, segments))
        if zeros_size > 0:
            big_segments = big_segments[:-zeros_size, :, :, :]

        predicts = np.zeros_like(self.training_labels, dtype=np.float32)
        for i in range(big_segments.shape[0]):
            a = big_segments[i, :, :, :, :]
            b = np.zeros_like(a, dtype=np.float32)
            b[a > 0.5] = 1.0
            predicts[i:i+self.seq_len, :, :, :] += b
        predicts[predicts > 0] = 1

        frames = np.repeat(self.training_frames, self.segment_groups, -1)
        segmented_frames = predicts * frames
        np.savez_compressed('/media/newhd/Ha/my_env/cell_5_segmentation/events_segments.npz',
                            predicts=predicts, predicts_frames=segmented_frames)




