import sys
import copy, time
import itertools
import os
import shutil
import sys
from inspect import getsourcefile

import tensorflow as tf
import numpy as np
from M3 import M3

# raw_input returns the empty string for "enter"
yes = {'y'}
no = {'n'}

reset = False  # Clear all learned models
sys.stdout.write('Do you want to reset training process? "y" or "n": ')
choice = raw_input().lower()
if choice in yes:
    reset = True
elif choice in no:
    reset = False
else:
    sys.stdout.write("Please respond with 'y' or 'n'")

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

# tensorboard --logdir=/media/newhd/Ha/my_env/cell_10_k/model_k_2/summary

# PROJECT_DIR = '/Users/phanha/Google Drive/Work/PyCharm/el_2.1'
PROJECT_DIR = '/media/newhd/Ha/my_env/cell_10_k'

seq_len = 12

MODEL_DIR = os.path.join(PROJECT_DIR, "model_k_" + str(seq_len))
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
SUMMARY_DIR = os.path.join(MODEL_DIR, "summary")

latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)

# Optionally empty model directory
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if reset:  # and latest_checkpoint is None:
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(SUMMARY_DIR, ignore_errors=True)
    latest_checkpoint = None

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

summary_writer = tf.summary.FileWriter(SUMMARY_DIR)

# optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
# sess = tf.InteractiveSession()

saver = None


# Get data
patch_size = 8
compress_jump = 2
def condense_raw(raw_tensor, compress_jump=compress_jump, patch_size=patch_size):
    uncompressed_patch_size = patch_size * compress_jump
    x_crop = raw_tensor.shape[2] - raw_tensor.shape[2] % uncompressed_patch_size
    y_crop = raw_tensor.shape[1] - raw_tensor.shape[1] % uncompressed_patch_size
    raw_tensor = raw_tensor[:, :y_crop, :x_crop]
    raw_tensor = raw_tensor[:, 0::compress_jump, 0::compress_jump]  # .astype(np.float32)
    return raw_tensor

data = np.load(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'raw_sequence', 'raw_sequence.npz'))
# raw_tensor = condense_raw(data['sequence'])
raw_tensor = data['sequence']


def read_ground_truth(file_path):
    """
    :param file_path:
    :return: [frame_id, x, y]
    """
    with open(file_path) as f:
        lines = f.readlines()
    # output = np.zeros((len(lines), 3))
    output = None
    skip = 0  # 111
    for i in range(len(lines)):
        content = lines[i].split(' ')
        content = [int(np.round(float(x))) for x in content]
        if int(float(content[0])) >= skip:
            # print content[0]
            if output is None:
                a = np.array(content[:3], dtype='float32')
                a[0] = a[0] - skip
                a[0] -= 1
                output = a
            else:
                a = np.array(content[:3], dtype='float32')
                a[0] = a[0] - skip
                a[0] -= 1
                output = np.vstack((output, a))
    # REORDER
    output_2 = np.zeros_like(output)
    output_2[:, 0] = output[:, 0]
    output_2[:, 1] = output[:, 2]
    output_2[:, 2] = output[:, 1]
    return output_2

gt_list = read_ground_truth(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'BAEC_seq5_mitosis.txt'))


start_time = time.time()
model = M3(sess, optimizer, saver, CHECKPOINT_DIR,
                 summary_writer=summary_writer,
                 training_data=raw_tensor,
                 training_labels=gt_list,
              seq_len=seq_len)
sess.run(tf.global_variables_initializer())
duration = time.time() - start_time
print "%.2f" % duration, 'graph building time ---'

saver_variables = tf.global_variables()
saver = tf.train.Saver(var_list=saver_variables, max_to_keep=4)
model.saver = saver


if latest_checkpoint is not None:
    # Load a previous checkpoint if it exists
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(latest_checkpoint + ".meta")
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    print tf.train.latest_checkpoint(CHECKPOINT_DIR), "tf.train.latest_checkpoint('./')"
    imported_meta.restore(sess, latest_checkpoint)

    if False:
        data = np.load('/media/newhd/Ha/data/BAEC/big/test.npz')
        model.test()
    else:
        model.run_train()
else:
    sess.graph.finalize()
    model.run_train()

