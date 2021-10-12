import os
import math
from os import listdir
from os.path import isfile, join

from scipy import misc
import numpy as np
import copy, collections
from scipy import ndimage

max_z = 210; max_x = 1392; max_y = 1040

base_url = ''
folder_url = '/media/newhd/Ha/data/BAEC_2'

img_dir = base_url + '/raw_img'
seq_dir = base_url + '/raw_sequence'
save_dir = base_url + '/processed/'
big_save_dir = base_url + '/big/'
gt_file = None
video = ''


def main():
    global folder_url, base_url, img_dir, seq_dir, save_dir, big_save_dir, gt_file, video
    folder_url = '/media/newhd/Ha/data/BAEC_2'
    videos = ['F0005']  # , 'F0002', 'F0003', 'F0004', 'F0005']
    video_gt = {
        'F0001': 'BAEC_seq1_mitosis.txt',
        'F0002': 'BAEC_seq2_mitosis.txt',
        'F0003': 'BAEC_seq3_mitosis.txt',
        'F0004': 'BAEC_seq4_mitosis.txt',
        'F0005': 'BAEC_seq5_mitosis.txt',
    }
    for video in videos:
        base_url = os.path.join(folder_url, video)
        img_dir = base_url + '/raw_img'
        seq_dir = base_url + '/raw_sequence'
        save_dir = folder_url + '/processed/'
        big_save_dir = folder_url + '/big/'

        gt_file = os.path.join(base_url, video_gt[video])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        if not os.path.exists(big_save_dir):
            os.makedirs(big_save_dir)

        if not os.path.isfile(join(seq_dir, 'raw_sequence.npz')):
            read_sequence_images(img_dir)
        # read_ground_truth(base_url+'/BAEC_seq1_mitosis.txt',
        #                                     base_url+'/divbi-classification',
        #                                     save_file=False)

        data_generator = DataGenerator(join(base_url + '/raw_sequence', 'raw_sequence.npz'))
        data_generator.generate_training_data()
        # data_generator.generate_candidate_sequences()
        # data_generator.generate_flips(type='_training_')
        # data_generator.generate_flips(type='_test_')
        if video == videos[-1] and False:
            data_generator.generate_final_files(folder_url+'/processed', folder_url+'/big')
            # data_generator.generate_ordered_sequence()
            data_generator.generate_progress_check_file(join(big_save_dir, 'train.npz'), big_save_dir)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    # ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    # return np.around(ndar, decimals=4)
    return ndar


def read_sequence_images(directory):
    file_list = [f for f in listdir(directory) if isfile(join(directory, f))]
    img_tensor = None; img_tensor_big = None
    img_list = []
    for f in file_list:
        img_ind = f.split('.')
        img_list.append(int(img_ind[0]))
    img_list.sort()
    for ind in img_list:
        if ind <= max_z:
            f = "%03d" % (ind)
            f += '.tif'
            print f
            img = np.array(misc.imread(join(directory, f))).astype(np.float32)
            img = scale_to_unit_interval(img)
            img = np.around(img, decimals=3)
            print np.amax(img), np.amin(img)
            img = np.array([img])
            if img_tensor is None:
                img_tensor = img
            else:
                img_tensor = np.vstack((img_tensor, img))

            if img_tensor.shape[0] >= 40:
                img_tensor_big = np.vstack((img_tensor_big, img_tensor)) if img_tensor_big is not None else img_tensor
                img_tensor = None

    img_tensor = np.vstack((img_tensor_big, img_tensor)) if img_tensor is not None else img_tensor_big
    print img_tensor.shape
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    np.savez_compressed(join(seq_dir, 'raw_sequence.npz'), sequence=img_tensor)
    """
    img_tensor = None
    for ind in img_list:
        if ind > len(img_list) - 100:
            f = "%03d" % (ind)
            f += '.tif'
            print f
            img = np.array(misc.imread(join(directory, f))).astype(np.float32)
            img = scale_to_unit_interval(img)
            img = np.around(img, decimals=3)
            print np.amax(img), np.amin(img)
            img = np.array([img])
            if img_tensor is None:
                img_tensor = img
            else:
                img_tensor = np.vstack((img_tensor, img))
    print img_tensor.shape
    if not os.path.exists(join(seq_dir, 'end')):
        os.makedirs(join(seq_dir, 'end'))
    np.savez_compressed(join(join(seq_dir, 'end'), 'raw_sequence_end.npz'), sequence=img_tensor)
    """


def read_ground_truth(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    # output = np.zeros((len(lines), 3))
    output = None
    skip = 0  # 111
    for i in range(len(lines)):
        content = lines[i].split(' ')
        content = [np.round(float(x)) for x in content]
        if int(float(content[0])) >= skip:
            print content[0]
            if output is None:
                a = np.array(content[:3], dtype='float32')
                a[0] = a[0] - skip
                output = a
            else:
                a = np.array(content[:3], dtype='float32')
                a[0] = a[0] - skip
                output = np.vstack((output, a))
    return output


def ground_truth_matrix(patch_size, compress_jump):
    groundtruth = read_ground_truth(gt_file)

    matrix = np.zeros((max_z, max_y, max_x))
    for i in range(groundtruth.shape[0]):
        z = int(groundtruth[i, 0] - 1)
        x = int(groundtruth[i, 1])
        y = int(groundtruth[i, 2])
        z0 = z - 1 if z - 1 > 0 else 0
        z1 = z + 2 if z + 2 < max_z else max_z
        x0 = x - 12 if x - 12 >= 0 else 0
        x1 = x + 13 if x + 13 <= max_x else max_x
        y0 = y - 12 if y - 12 >= 0 else 0
        y1 = y + 13 if y + 13 <= max_y else max_y
        z0 = int(z0); z1=int(z1); y0=int(y0); y1=int(y1); x0=int(x0); x1=int(x1)
        print z0, z1, y0, y1, x0, x1
        # matrix[z0:z1, y0:y1, x0:x1] = 1
        matrix[z, y:y+compress_jump, x:x+compress_jump] = 1
    """
    if False:
        x_crop = matrix.shape[2] - matrix.shape[2] % (patch_size * compress_jump)
        y_crop = matrix.shape[1] - matrix.shape[1] % (patch_size * compress_jump)
        matrix = matrix[:, :y_crop, :x_crop]
        matrix = matrix[:, 0::compress_jump, 0::compress_jump]
        # np.savez(os.path.join(big_save_dir, 'ground_truth_matrix.npz'), groundtruth=matrix)
    """
    return matrix





class DataGenerator(object):
    def __init__(self, path_begin, path_end=''):
        self.path_begin = path_begin
        # self.path_end = path_end
        self.seq_length = 10
        self.compress_jump = 2
        self.patch_size = 20
        self.patch_step = self.patch_size / 2
        self.uncompressed_patch_size = self.patch_size * self.compress_jump
        # self.frame_jump = 10

        self.gt_matrix = ground_truth_matrix(self.patch_size, self.compress_jump)

        self.train_percentage = 0.90
        self.valid_percentage = 0.0
        self.test_percentage = 0.10

    def condense_raw(self):
        global max_z, max_x, max_y
        x_crop = self.raw_tensor.shape[2] - self.raw_tensor.shape[2] % self.uncompressed_patch_size
        y_crop = self.raw_tensor.shape[1] - self.raw_tensor.shape[1] % self.uncompressed_patch_size
        self.raw_tensor = self.raw_tensor[:, :y_crop, :x_crop]
        self.raw_tensor = self.raw_tensor[:, 0::self.compress_jump, 0::self.compress_jump]  # .astype(np.float32)
        max_z, max_x, max_y = self.raw_tensor.shape[0], self.raw_tensor.shape[2], self.raw_tensor.shape[1]
        print self.raw_tensor.shape

    def get_gt_coord(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        upper_limit = max_z
        coords = []
        for i in range(len(lines)):
            content = lines[i].split(' ')
            content = [np.round(float(x)) for x in content]
            if int(float(content[0])) <= upper_limit and content[2] <= max_y * self.compress_jump and content[1] <= max_x * self.compress_jump:
                a = np.array(content[:3], dtype='float32')
                coords.append(a)
        return coords

    def generate_training_data(self):
        data = np.load(self.path_begin)
        self.raw_tensor = data['sequence']
        self.condense_raw()
        gt_coords = self.get_gt_coord(gt_file)

        big_frame_crops = None; frame_crops = None; frame_counts = []
        for z in range(self.raw_tensor.shape[0]):
            print z
            crops, event_count = self.generate_crops(z+1, self.raw_tensor[z, :, :], gt_coords)
            # crops = crops[:, ::2, ::2]
            crops = np.array([crops])
            if frame_crops is None:
                frame_crops = crops
            else:
                frame_crops = np.vstack((frame_crops, crops))

            if frame_crops.shape[0] >= 20:
                big_frame_crops = frame_crops if big_frame_crops is None else np.vstack((big_frame_crops, frame_crops))
                frame_crops = None
            frame_counts.append(event_count)
        frame_crops = big_frame_crops if frame_crops is None else np.vstack((big_frame_crops, frame_crops))
        print frame_counts, sum(frame_counts)
        print frame_crops.shape
        # frame_crops: [frame_id, crops_count, patch_size, patch_size]
        # frame_counts: [frame_id, count]
        np.savez_compressed(join(save_dir, video + '_training_' + '_normal_seq.npz'), frame_crops=frame_crops, frame_counts=frame_counts)
        np.savez_compressed(join(big_save_dir, 'progress.npz'), input_raw_data=frame_crops[0:1, :20, :, :])



    def generate_crops(self, frame_id, frame, gt_coords):
        x = 0
        y = 0
        pad = 4
        crop_count = ((frame.shape[0] - self.patch_size) // self.patch_step) * ((frame.shape[1] - self.patch_size) // self.patch_step)
        crops = np.zeros([crop_count * 2, self.patch_size + pad*2, self.patch_size + pad*2])
        # crops = np.zeros([crop_count * 2, self.patch_size, self.patch_size])
        count = 0
        while (x + self.patch_size <= frame.shape[1]):
            y = 0
            while (y + self.patch_size <= frame.shape[0]):

                if y-pad < 0:
                    y = pad
                if x-pad < 0:
                    x = pad
                if y+self.patch_size+pad > frame.shape[0]:
                    y = frame.shape[0] - self.patch_size - pad
                if x+self.patch_size+pad > frame.shape[1]:
                    x = frame.shape[1] - self.patch_size - pad
                crop = frame[y-pad:y+self.patch_size+pad, x-pad:x+self.patch_size+pad]

                # crop = frame[y:y+self.patch_size, x:x+self.patch_size]
                crops[count, :, :] = crop
                count += 1
                y += self.patch_step * 2
            # print count, x, y, crop_count
            x += self.patch_step * 2
        crops = crops[:count, :, :]
        count = 0
        for coords in gt_coords:
            if coords[0] == frame_id:
                count += 1
        return crops, count


    def generate_final_files(self, file_path, save_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        # file_list.reverse()
        big_input_raw_data = None; big_output_raw_data = None
        big_input_raw_data_test = None; big_output_raw_data_test = None
        print file_list
        for file in file_list:
            if '_training_' in file:
                if big_input_raw_data is None or big_input_raw_data.shape[0] < 50000:
                    print file
                    data = np.load(os.path.join(file_path, file))
                    input_raw_data = data['input_raw_data']  # np.expand_dims(data['input_raw_data'], axis=1)
                    output_raw_data = data['output_raw_data']  # np.expand_dims(data['output_raw_data'], axis=1)

                    # input_raw_data = scale_to_unit_interval(input_raw_data)
                    print input_raw_data.shape, output_raw_data.shape
                    if big_input_raw_data is None:
                        big_input_raw_data = input_raw_data
                    else:
                        print big_input_raw_data.shape
                        big_input_raw_data = np.vstack((big_input_raw_data, input_raw_data))

                    if big_output_raw_data is None:
                        big_output_raw_data = output_raw_data
                    else:
                        print big_output_raw_data.shape, output_raw_data.shape, 'big_output_raw_data, output_raw_data'
                        big_output_raw_data = np.vstack((big_output_raw_data, output_raw_data))
            elif '_test_' in file:
                if big_input_raw_data_test is None or big_input_raw_data_test.shape[0] < 50000:
                    print file
                    data = np.load(os.path.join(file_path, file))
                    input_raw_data = data['input_raw_data']  # np.expand_dims(data['input_raw_data'], axis=1)
                    output_raw_data = data['output_raw_data']  # np.expand_dims(data['output_raw_data'], axis=1)

                    # input_raw_data = scale_to_unit_interval(input_raw_data)
                    print input_raw_data.shape, output_raw_data.shape
                    if big_input_raw_data_test is None:
                        big_input_raw_data_test = input_raw_data
                    else:
                        print big_input_raw_data_test.shape
                        big_input_raw_data_test = np.vstack((big_input_raw_data_test, input_raw_data))

                    if big_output_raw_data_test is None:
                        big_output_raw_data_test = output_raw_data
                    else:
                        print big_output_raw_data_test.shape, output_raw_data.shape, 'big_output_raw_data, output_raw_data'
                        big_output_raw_data_test = np.vstack((big_output_raw_data_test, output_raw_data))

        order = np.arange(big_input_raw_data.shape[0])
        np.random.shuffle(order)
        big_input_raw_data = big_input_raw_data[order, :, :, :]
        big_output_raw_data = big_output_raw_data[order, :, :, :]

        order = np.arange(big_input_raw_data_test.shape[0])
        np.random.shuffle(order)
        big_input_raw_data_test = big_input_raw_data_test[order, :, :, :]
        big_output_raw_data_test = big_output_raw_data_test[order, :, :, :]

        big_input_raw_data = big_input_raw_data[:30000, :, :, :]  # 19052 samples
        big_output_raw_data = big_output_raw_data[:30000, :, :, :]
        big_input_raw_data_test = big_input_raw_data_test[:30000, :, :, :]  # 19052 samples
        big_output_raw_data_test = big_output_raw_data_test[:30000, :, :, :]

        np.savez_compressed(os.path.join(save_path, 'train.npz'), input_raw_data=big_input_raw_data, output_raw_data=big_output_raw_data)
        np.savez_compressed(os.path.join(save_path, 'test.npz'), input_raw_data=big_input_raw_data_test, output_raw_data=big_output_raw_data_test)

    def generate_progress_check_file(self, file_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = np.load(file_path)
        seq = data['input_raw_data']
        seq_output = data['output_raw_data']

        input_raw_data = seq[:16, :, :, :]
        output_raw_data = seq_output[:16, :, :, :]

        big_file_path = os.path.join(save_path, 'progress.npz')
        np.savez_compressed(big_file_path, input_raw_data=input_raw_data, output_raw_data=output_raw_data)


# read_sequence_images('/home/ha/theano/data/cell_division/F0001_input_output/raw_img')
# read_ground_truth(base_url+'/BAEC_seq1_mitosis.txt',
#                                     base_url+'/divbi-classification',
#                                     save_file=False)

# data_generator = DataGenerator(join(base_url+'/raw_sequence','raw_sequence_begin.npz'),
#                                join(base_url+'/raw_sequence','end/raw_sequence_end.npz'))
# data_generator.generate_candidate_sequences()
# data_generator.generate_normal()
# data_generator.generate_flips()

# data_generator.generate_final_files(base_url+'/processed', base_url+'/big')

# data_generator.generate_ordered_sequence()
# data_generator.generate_progress_check_file(join(big_save_dir, 'train.npz'), base_url+'/big')

main()