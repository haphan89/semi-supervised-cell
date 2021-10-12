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

# tensorboard --logdir=/media/newhd/Ha/my_env/cell_5_segmentation_F0005/model_postprocess/summary
# 10.66.31.24:6006
# hostname -I  # Check hostname
# sudo netstat -ntlp | grep LISTEN  # Check ports open

# PROJECT_DIR = '/Users/phanha/Google Drive/Work/PyCharm/el_2.1'
PROJECT_DIR = '/media/newhd/Ha/my_env/cell_5_segmentation_F0005'

MODEL_DIR = os.path.join(PROJECT_DIR, "model_postprocess")
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

# GET DATA
data = np.load(os.path.join('/media/newhd/Ha/my_env/cell_5_segmentation_F0005', 'clean_segments.npz'))
training_data = data['segmentation'][:, ::2, ::2]
training_data = np.expand_dims(training_data, -1)

data = np.load(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'raw_sequence', 'raw_sequence.npz'))
raw_tensor = data['sequence'][:, ::4, ::4]
raw_tensor = np.expand_dims(raw_tensor, -1)


saver = None

start_time = time.time()
model = M3(sess, optimizer, saver, CHECKPOINT_DIR,
                 summary_writer=summary_writer,
                 training_data=training_data,
                 training_frames=raw_tensor)
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

    if True:
        ends_segments = np.load(os.path.join('/media/newhd/Ha/my_env/cell_5_segmentation_F0005', 'ends_segments.npz'))
        ends_segments = ends_segments['map'][:, ::2, ::2]
        ends_segments = np.expand_dims(ends_segments, -1)
        model.inference(ends_segments)
    else:
        model.run_train()
else:
    sess.graph.finalize()
    model.run_train()



"""
sudo apt-get update
sudo apt-get install --no-install-recommends nvidia-384 libcuda1-384 nvidia-opencl-icd-384
sudo reboot
"""