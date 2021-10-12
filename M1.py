import tensorflow as tf
import numpy as np
import os, io
import time

class M1(object):
    def __init__(self, session,
                 optimizer,
                 saver,
                 checkpoint_dir,
                 # max_gradient=5,
                 summary_writer=None,
                 summary_every=100,
                 save_every=2000,
                 training=True,
                 size_x=64,
                 size_y=64,
                 seq_len=1271,
                 no_cuts=1,
                 batch_size=8,
                 training_data=None,
                 training_labels=None,
                 training_frames=None,
                 patch_size=12):
        self.session = session
        self.optimizer = optimizer
        self.saver = saver
        # self.max_gradient = max_gradient
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.training = training

        self.pool_size = patch_size
        self.segment_groups = 6

        self.batch_size = batch_size
        self.no_cuts = no_cuts
        self.size_x = training_frames.shape[1]
        self.size_y = training_frames.shape[2]
        self.training_frames = training_frames  # frames: [frame_id, x, y, 1]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.create_variables()
        self.summary_writer.add_graph(self.session.graph)

        self.compress_jump = 1
        self.z_tolerance = 1
        self.xy_tolerance = 10
        self.patch_size = patch_size

    def create_variables(self):
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, 1], name='input')
        # self.input_1 = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, 1], name='input_1')
        self.masks_dropout = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, 1], name='masks')
        self.one_out_masks = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, self.segment_groups], name='one_out_masks')
        self.keep_one_masks = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, self.segment_groups], name='keep_one_masks')
        self.keep_one_inds = tf.placeholder(tf.int32, [self.batch_size, ], name='keep_one_inds')
        # self.one_out_masks_2 = tf.placeholder(tf.float32, [self.batch_size, self.size_x, self.size_y, self.segment_groups], name='one_out_masks_2')

        self.loss_dropout, self.loss_reconstr, self.loss_probs_diversity, self.loss_probs_asso = self.model()
        # self.loss_probs *= self.segment_groups

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'bias' not in v.name]) * 1e-6

        # self.loss = self.loss_dropout + self.loss_reconstr  # + self.loss_probs_diversity
        self.loss = self.loss_reconstr + self.lossL2 + self.loss_dropout  # + self.loss_probs_asso

        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)

        ones = tf.ones_like(self.input)
        a = self.probs[0:1, :, :, :]
        a = tf.where(tf.greater(a, 0.0),
                     tf.constant(1.0, shape=a.shape, dtype=tf.float32),
                     tf.constant(0.0, shape=a.shape, dtype=tf.float32))
        probs_images = tf.concat((self.input[0:1, :, :, :],
                                  tf.transpose(self.probs[0:1, :, :, :], [3, 1, 2, 0])), 0)
                                  # tf.transpose(self.input[0:1, :, :, :] * a, [3, 1, 2, 0])), 0)

        kernel_tensor = tf.constant(0.5, shape=[3, 3, self.segment_groups])
        probs_erosion = tf.nn.erosion2d(self.probs[0:1, :, :, :], kernel=kernel_tensor, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        probs_erosion_mean = tf.reduce_mean(probs_erosion)
        probs_erosion = tf.where(tf.greater(probs_erosion, probs_erosion_mean),
                     tf.constant(1.0, shape=probs_erosion.shape, dtype=tf.float32),
                     tf.constant(0.0, shape=probs_erosion.shape, dtype=tf.float32))
        probs_erosion_images = tf.concat((self.input[0:1, :, :, :],
                                  tf.transpose(probs_erosion, [3, 1, 2, 0])), 0)

        probs_dropout_images = tf.concat((self.input[0:1, :, :, :],
                                          tf.transpose(self.dropout_reconstr[0:1, :, :, :], [3, 1, 2, 0])), 0)

        # probs_asso_predict_show = tf.concat((self.input[0:1, :, :, :],
        #                                   tf.transpose(self.probs_asso_predict_show[0:1, :, :, :], [3, 1, 2, 0])), 0)


        """
        ones2 = tf.ones_like(self.probs[0:1, :, :, :])
        max_prob = tf.reduce_max(self.probs[0:1, :, :, :])
        probs_reconstr_images = tf.concat((tf.transpose(self.probs[0:1, :, :, :], [3, 1, 2, 0]) / max_prob,
                                           tf.transpose(ones2, [3, 1, 2, 0])[:, :, 0:1, :],
                                           tf.transpose(self.probs_reconstr[0:1, :, :, :], [3, 1, 2, 0]) / max_prob), 2)
        """

        """
        dropout_reconstr_images = tf.concat((self.input,
                                           ones[:, :, 0:1, :],
                                           self.dropout_reconstr), 2)
        """

        input_reconstr_images = tf.concat((self.input,
                                           ones[:, :, 0:1, :],
                                           self.input_reconstr), 2)

        """
        # kernel_tensor = tf.constant(0.5, shape=[2, 2, self.segment_groups])
        # self.probs = tf.nn.erosion2d(self.probs, kernel=kernel_tensor, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        a = self.probs[0:1, :, :, :]
        a = tf.where(tf.greater(a, 0.0),
                     tf.constant(1.0, shape=a.shape, dtype=tf.float32),
                     tf.constant(0.0, shape=a.shape, dtype=tf.float32))
        b = self.probs_1[0:1, :, :, :]
        b = tf.where(tf.greater(b, 0.0),
                     tf.constant(1.0, shape=b.shape, dtype=tf.float32),
                     tf.constant(0.0, shape=b.shape, dtype=tf.float32))
        # probs_diff = self.probs[0:1, :, :, :] - self.probs_1[0:1, :, :, :]
        # probs_diff = probs_diff / tf.reduce_max(probs_diff)
        probs_a = a * tf.tile(self.input[0:1, :, :, :], [1, 1, 1, self.segment_groups])
        probs_b = a * tf.tile(self.input_1[0:1, :, :, :], [1, 1, 1, self.segment_groups])
        probs_diff_images = tf.concat((
            tf.concat((self.input[0:1, :, :, :], ones[0:1, :, 0:1, :], self.input_1[0:1, :, :, :]), 2),
            tf.concat((tf.transpose(probs_a, [3, 1, 2, 0]), ones[:self.segment_groups, :, 0:1, :], tf.transpose(probs_b, [3, 1, 2, 0])), 2)), 0)
        """

        probs_images_sm = tf.summary.image("probs", probs_images, self.segment_groups + 1)
        probs_erosion_images_sm = tf.summary.image("probs_erosion", probs_erosion_images, self.segment_groups + 1)
        probs_dropout_images_sm = tf.summary.image("probs_dropout", probs_dropout_images, self.segment_groups + 1)
        # probs_reconstr_images_sm = tf.summary.image("dropout_reconstr", dropout_reconstr_images, self.batch_size)
        input_reconstr_images_sm = tf.summary.image("input_reconstr", input_reconstr_images, self.batch_size)
        # probs_diff_images_sm = tf.summary.image("probs_diff", probs_diff_images, self.batch_size)
        # probs_asso_images_sm = tf.summary.image("probs_asso_images", probs_asso_predict_show, self.segment_groups + 1)

        """
        asso_softmax = tf.expand_dims(self.asso_softmax, axis=0)
        asso_softmax = tf.expand_dims(asso_softmax, axis=-1)
        ones = tf.ones_like(asso_softmax)
        asso_softmax = tf.concat((ones[:, :, 0:1, :], asso_softmax), 2)
        probs_asso_sm = tf.summary.image("probs_asso", asso_softmax, 1)
        """

        cost_sm = tf.summary.scalar("cost", self.loss)
        cost_dropout_sm = tf.summary.scalar("cost_dropout_reconstr", self.loss_dropout)
        cost_reconstr_sm = tf.summary.scalar("cost_input_reconstr", self.loss_reconstr)
        cost_diversity_sm = tf.summary.scalar("cost_probs_diversity", self.loss_probs_diversity)
        cost_probs_asso_sm = tf.summary.scalar("cost_probs_asso_diversity", self.loss_probs_asso)
        cost_l2_sm = tf.summary.scalar("cost_l2", self.lossL2)
        self.merge_list = [probs_images_sm, probs_dropout_images_sm, probs_erosion_images_sm, # probs_reconstr_images_sm,
                           input_reconstr_images_sm, # probs_asso_images_sm, probs_asso_sm,
                           cost_sm, cost_dropout_sm, cost_reconstr_sm, cost_diversity_sm, cost_probs_asso_sm,
                           cost_l2_sm]
        self.summarize = tf.summary.merge(self.merge_list)

    def model(self):
        with tf.variable_scope('segmentation'):
            self.logits, self.probs, self.dropout_reconstr, self.input_reconstr, self.probs_dropout = self.segmentation(self.input)
            # self.probs_asso_predict, self.maps_asso_scores = self.segmentation(self.input)

            # self.probs_asso_predict_show = self.probs_asso_predict * self.maps_asso_scores
        # with tf.variable_scope('segmentation', reuse=True):
        #     _, self.probs_1, _, _, _ = self.segmentation(self.input_1)

        # loss_probs = tf.losses.mean_squared_error(labels=self.probs, predictions=self.probs_reconstr)
        loss_reconstr = tf.losses.mean_squared_error(labels=self.input, predictions=self.input_reconstr)

        # loss_probs = tf.reduce_mean(tf.square(self.probs - self.probs_reconstr)) * 1e1  # / (tf.reduce_sum(self.probs, [1, 2]) + 1))
        loss_dropout = tf.losses.mean_squared_error(labels=self.input, predictions=self.dropout_reconstr)
        # loss_dropout = tf.constant(0.0)

        # mean_probs_class = tf.reduce_mean(self.probs, [1, 2])
        # loss_probs_diversity = tf.reduce_mean(tf.square(mean_probs_class - tf.reduce_mean(mean_probs_class, -1, keepdims=True)))
        loss_probs_diversity = tf.constant(0.0)

        loss_probs_asso = tf.constant(0.0)  # tf.reduce_mean(self.maps_asso_scores * tf.square(self.probs_asso_predict - self.probs)) * 40

        return loss_dropout, loss_reconstr, loss_probs_diversity, loss_probs_asso

    def segmentation(self, input):
        enc = input
        filters = [24, 24, self.segment_groups]
        for i in range(len(filters)):
            enc = tf.layers.conv2d(enc, filters=filters[i], kernel_size=5, strides=(1, 1), padding='same')
                                   # activation=tf.nn.relu)
            # enc = tf.layers.dropout(enc, rate=0.1)

        logits = enc  # tf.layers.dense(enc, self.segment_groups)
        probs = tf.nn.softmax(logits, -1)

        max_mask = tf.where(tf.equal(tf.reduce_max(probs, axis=-1, keepdims=True), probs),
                            tf.constant(1.0, shape=probs.shape, dtype=tf.float32),
                            tf.constant(0.0, shape=probs.shape, dtype=tf.float32))
        probs = probs * max_mask


        positive_mask = tf.where(tf.greater(probs, 0.0),
                                 tf.constant(1.0, shape=probs.shape, dtype=tf.float32),
                                 tf.constant(0.0, shape=probs.shape, dtype=tf.float32))

        # negative_mask = tf.where(tf.equal(probs, 0.0),
        #                          tf.constant(1.0, shape=probs.shape, dtype=tf.float32),
        #                          tf.constant(0.0, shape=probs.shape, dtype=tf.float32))
        # probs1 = probs / (probs + negative_mask)

        # kernel_tensor = tf.constant(0.5, shape=[2, 2, self.segment_groups])
        # probs_eroded = tf.nn.erosion2d(probs, kernel=kernel_tensor, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')

        # probs1 = probs * self.one_out_masks_2 + probs_eroded * (1.0 - self.one_out_masks_2)

        """
        # Segment association --------------------------------------------------

        probs_2 = probs * self.keep_one_masks
        diag = tf.diag(tf.ones(self.segment_groups, dtype=tf.float32))

        initializer = tf.contrib.layers.xavier_initializer()
        # self.asso_pre_softmax_storage = tf.get_variable("map_asso_pre_softmax", [self.segment_groups, self.segment_groups],
        #                                     dtype=tf.float32, initializer=initializer, trainable=True)
        ones = tf.ones([1, self.segment_groups * self.segment_groups])
        self.asso_pre_softmax_storage = tf.layers.dense(ones, self.segment_groups * self.segment_groups)
        self.asso_pre_softmax_storage = tf.layers.dense(self.asso_pre_softmax_storage, self.segment_groups * self.segment_groups)
        self.asso_pre_softmax_storage = tf.reshape(self.asso_pre_softmax_storage, [self.segment_groups, self.segment_groups])

        self.asso_pre_softmax = tf.abs(self.asso_pre_softmax_storage)
        self.asso_pre_softmax = self.asso_pre_softmax * (1.0 - diag)
        self.asso_softmax = tf.nn.softmax(self.asso_pre_softmax, -1)
        self.asso_softmax = self.asso_softmax * (1.0 - diag)
        self.asso_softmax += tf.transpose(self.asso_softmax)

        asso_softmax = tf.expand_dims(self.asso_softmax, axis=0)
        asso_softmax = tf.tile(asso_softmax, multiples=[self.batch_size, 1, 1])

        a = self.keep_one_masks[:, 0, 0, :]
        a = tf.expand_dims(a, axis=-1)
        a = tf.tile(a, multiples=[1, 1, self.segment_groups])
        asso_softmax = a * asso_softmax
        asso_softmax = tf.reduce_sum(asso_softmax, axis=1)

        asso_softmax = tf.expand_dims(asso_softmax, axis=1)
        asso_softmax = tf.expand_dims(asso_softmax, axis=1)
        maps_asso = tf.tile(asso_softmax, multiples=[1, self.size_x, self.size_y, 1])

        enc = probs_2  # maps_reconstr
        filters = [24, 24, self.segment_groups]
        for i in range(len(filters)):
            enc = tf.layers.conv2d(enc, filters=filters[i], kernel_size=5, strides=(1, 1), padding='same',
                                   activation=tf.nn.relu)
        probs_predict = enc
        """


        # RECONSTRUCTION =====================================================================================

        probs1 = probs
        # probs1 = probs1 * positive_mask
        # probs1 = (probs1 + tf.clip_by_value(tf.random_uniform(probs1.shape, minval=-0.2, maxval=0.8), -0.2, 0.8)) * positive_mask
        probs1 = (probs1 + tf.random_uniform(probs1.shape, minval=-0.2, maxval=0.8)) * positive_mask
        # positive_mask = probs * 1.1

        # probs1 = probs / (probs + negative_mask)

        # probs = tf.clip_by_value(probs, 0, 0.1 + 1.0/self.segment_groups)

        probs1 = probs1 * self.one_out_masks

        masks = tf.tile(self.masks_dropout, [1, 1, 1, self.segment_groups])
        probs_dropout = probs * masks
        enc = probs_dropout

        filters = [24, 24, 1]
        for i in range(len(filters)):
            enc = tf.layers.conv2d(enc, filters=filters[i], kernel_size=5, strides=(1, 1), padding='same',
                                   activation=tf.nn.relu)
        # enc = tf.reduce_sum(enc * positive_mask, -1, keepdims=True)

        dropout_reconstr = enc

        # enc = probs1  # * tf.random_uniform(probs.shape, minval=0.1, maxval=10)
        enc = probs1  # dropout_reconstr

        filters = [24, 24, 1]  # self.segment_groups]
        for i in range(len(filters)):
            enc = tf.layers.conv2d(enc, filters=filters[i], kernel_size=5, strides=(1, 1), padding='same',
                                   activation=tf.nn.relu)
        # enc = tf.reduce_sum(enc * dropout_positive_mask, -1, keepdims=True)

        input_reconstr = enc

        return logits, probs, dropout_reconstr, input_reconstr, probs_dropout # , probs_predict, maps_asso
        # return logits, probs, input_reconstr

    def label_smoothing(self, input, epsilon=0.1):
        K = 2  # number of channels
        return ((1 - epsilon) * input) + (epsilon / K)

    def tf_repeat(self, tensor, repeats):
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
            repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tensor

    def expand_dots(self, masks):
        filter = tf.ones([self.pool_size, self.pool_size, 1, 1])
        masks_2 = tf.nn.conv2d_transpose(masks, filter=filter,
                                                output_shape=masks.shape,
                                                strides=[1, 1, 1, 1], padding='SAME')
        return masks_2

    def expand_tensor(self, masks):
        shape = tf.convert_to_tensor([masks.shape[0], masks.shape[1] * self.pool_size, masks.shape[2] * self.pool_size, masks.shape[3]], dtype=tf.int32)
        filter = tf.ones([self.pool_size, self.pool_size, 1, 1])
        masks_2 = tf.nn.conv2d_transpose(masks, filter=filter,
                                         output_shape=shape,
                                         strides=[1, self.pool_size, self.pool_size, 1], padding='SAME')
        return masks_2

    def get_sample(self):
        n = np.random.randint(0, self.training_frames.shape[0] - 6, size=self.batch_size)
        frames = self.training_frames[n, :, :, :]
        frames_1 = self.training_frames[n + 6, :, :, :]
        masks = self.random_dropout_masks(frames)

        one_out_masks = np.ones_like(frames)
        one_out_masks = np.repeat(one_out_masks, self.segment_groups, axis=-1)
        for i in range(one_out_masks.shape[0]):
            x = np.random.randint(0, self.segment_groups, 1)
            one_out_masks[i, :, :, x] = 0.0

        one_out_masks_2 = np.ones_like(frames)
        one_out_masks_2 = np.repeat(one_out_masks_2, self.segment_groups, axis=-1)
        for i in range(one_out_masks_2.shape[0]):
            x = np.random.randint(0, self.segment_groups, 1)
            one_out_masks_2[i, :, :, x] = 0.0

        # print one_out_masks[:, 0, 0, :], 'one_out_masks'

        keep_one_masks = np.zeros_like(frames)
        keep_one_masks = np.repeat(keep_one_masks, self.segment_groups, axis=-1)
        keep_one_inds = np.zeros(self.batch_size)
        for i in range(keep_one_masks.shape[0]):
            x = np.random.randint(0, self.segment_groups, 1)
            keep_one_masks[i, :, :, x] = 1.0
            keep_one_inds[i] = x

        return frames, frames_1, masks, one_out_masks, one_out_masks_2, keep_one_masks, keep_one_inds

    def get_inference_frames(self, x=0):
        more_frames = True
        frames = self.training_frames[x:x + self.batch_size, :, :, :]
        shape = frames.shape
        zeros_size = 0
        if shape[0] < self.batch_size:
            frames = np.concatenate((frames, np.zeros([self.batch_size - frames.shape[0]] + list(shape[1:]))), 0)
            more_frames = False
            zeros_size = self.batch_size - shape[0]
        return frames, x + self.batch_size, more_frames, zeros_size

    def random_dropout_masks(self, frames, window=6, dropout=0.2):
        masks = np.ones_like(frames)[:, ::window, ::window, :]
        ori_shape = masks.shape
        masks = np.reshape(masks, [masks.shape[0], -1])
        n = np.ones_like(masks)
        for i in range(masks.shape[0]):
            a = np.arange(masks.shape[1], dtype=np.int32)
            np.random.shuffle(a)
            inds = a[:np.floor(n.shape[1] * dropout).astype(np.int32)]
            masks[i, inds] = 0
            # n[i, :] = a
        # n = n[:, :np.floor(n.shape[1] * dropout).astype(np.int32)]
        # print n, 'n'
        masks[n.astype(np.int32)] = 0
        # print masks
        masks = np.reshape(masks, ori_shape)
        masks = np.repeat(masks, window, axis=1)
        masks = np.repeat(masks, window, axis=2)
        masks = masks[:, :frames.shape[1], :frames.shape[2]]
        # print masks.shape, frames.shape, 'masks.shape, frames.shape'
        return masks

    def run_train(self):
        with self.session.as_default(), self.session.graph.as_default():
            print 'started ---'
            self.gs = self.session.run(self.global_step)
            try:
                while self.gs <= 4000:
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
        frames, frames_1, masks, one_out_masks, one_out_masks_2, keep_one_masks, keep_one_inds = self.get_sample()

        feed_dict = {
            self.input: frames,
            # self.input_1: frames_1,
            self.masks_dropout: masks,
            self.one_out_masks: one_out_masks,
            self.keep_one_masks: keep_one_masks,
            self.keep_one_inds: keep_one_inds,
            # self.one_out_masks_2: one_out_masks_2
        }
        # p_z_grads, masks, diff_grads, \
        loss, summary, _, self.gs = self.session.run([
            self.loss, self.summarize, self.train_op, self.global_step], feed_dict)
        duration = time.time() - start_time

        """
        a = np.where(prob_labels[:, 1] > 0.5)[0]
        b = np.random.randint(0, prob_labels.shape[0], len(a))
        print np.transpose(prob_labels[:10, :]), 'prob_labels', np.sum(prob_labels, 1)
        print np.transpose(logits[a, :]), 'logits'
        print np.transpose(logits[b, :]), 'logits random'
        print np.transpose(np.reshape(probs, [self.batch_size * self.seq_len, -1])[a, :]), 'probs'
        """
        # emit summaries
        if self.gs % 5 == 1:
            print duration, self.gs, 'duration, self.gs'
            print loss, 'loss'

        if self.gs % 5 == 4:
            print 'summary ---'
            self.summary_writer.add_summary(summary, self.gs)

        if self.gs % 2000 == 100:
            print("Saving model checkpoint: {}".format(str(self.gs)))
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)

    def inference(self):
        more_frames = True
        x = 0
        big_probs = None
        zeros_size = 0
        while more_frames:
            print x, 'x'
            frames, x, more_frames, zeros_size = self.get_inference_frames(x)
            one_out_masks = np.ones_like(frames)
            one_out_masks = np.repeat(one_out_masks, self.segment_groups, axis=-1)
            masks = np.ones_like(frames)

            start_time = time.time()
            feed_dict = {
                self.input: frames,
                self.masks_dropout: masks,
                self.one_out_masks: one_out_masks,
            }

            probs = self.session.run(self.probs, feed_dict)
            duration = time.time() - start_time
            print duration, 'sec'

            probs[probs > 0] = 1.0
            big_probs = probs if big_probs is None else np.vstack((big_probs, probs))
        if zeros_size > 0:
            big_probs = big_probs[:-zeros_size, :, :, :]
        frames = np.repeat(self.training_frames, self.segment_groups, -1)
        segmented_frames = big_probs * frames
        np.savez_compressed('/media/newhd/Ha/my_env/cell_5_segmentation_F0005/segmentation.npz',
                            segmentation=big_probs, frames=self.training_frames)  #, segmented_frames=segmented_frames)




