import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.measure import label
from scipy.ndimage import label, binary_opening, binary_dilation, binary_erosion
from skimage import morphology
import os


class PostProcess(object):
    def __init__(self, base_url='', create_clean_segments=False, create_ends_segments=False, analyze_predicts=True,
                 average_cell_size=True):
        self.base_url = base_url
        if create_clean_segments:
            self.data = np.load(os.path.join(self.base_url, 'segmentation.npz'))
            self.segmentation = self.data['segmentation']
            self.frames = self.data['frames']
            print self.segmentation.shape, self.frames.shape, 'shapes'
            # self.show_images()

            self.segmentation = self.segmentation[:, :, :, 3]  # SELECTING the first class

            self.clean_segments()
        elif create_ends_segments:
            self.segmentation = np.load(os.path.join(self.base_url, 'clean_segments.npz'))
            self.segmentation = self.segmentation['segmentation']
            if False:
                fig = plt.figure()
                fig.patch.set_facecolor('white')
                seq = self.segmentation[:10, :, :]
                for i in range(seq.shape[0]):
                    a = fig.add_subplot(2, 5, i + 1)
                    imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')
                plt.show()

            self.find_ends_fast(self.segmentation)
            # self.show_ends()
        elif analyze_predicts:
            if True:
                self.segmentation = np.load(os.path.join(self.base_url, 'clean_segments.npz'))
                self.segmentation = self.segmentation['segmentation']
                self.predicts = np.load(os.path.join(self.base_url, 'events_prediction.npz'))
                self.predicts = self.predicts['predicts']
                self.analyze_predicts()
                # self.match_gt_segmentation()
            else:
                self.show_outside_segmentation_mitosis()
        elif average_cell_size:
            self.data = np.load(os.path.join(self.base_url, 'clean_segments.npz'))
            self.segmentation = self.data['segmentation']
            self.average_cell_size(self.segmentation)


    def show_images(self):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        seq = self.segmentation[0, :, :, :]
        for i in range(seq.shape[-1]):
            a = fig.add_subplot(2, 4, i + 1)
            imgplot = plt.imshow(seq[:, :, i], cmap='Greys_r')
        plt.show()

    def clean_segments(self):
        self.segmentation = binary_opening(self.segmentation, structure=np.ones((1, 5, 5))).astype(self.segmentation.dtype)
        self.label_segments()

    def label_segments(self, min_area=60):
        self.segmentation[self.segmentation > 0] = 1.0
        labels, number = label(self.segmentation)
        print number, 'number'

        # labels = labels[:10, :, :]

        if False:
            unique_labels = np.unique(labels)
            print unique_labels.shape, 'unique_labels'
            for a in unique_labels:
                if a > 0:
                    l = deepcopy(labels)
                    l[l != a] = 0
                    # l[l == a] = 1
                    area = np.sum(l) / a
                    if area < min_area:
                        m = np.ones_like(l)
                        m[l > 0] = 0
                        labels *= m
                    else:
                        print area, 'area', a, number
            np.savez_compressed(os.path.join(self.base_url, 'clean_segments.npz'), segmentation=labels)

        if True:
            labels = morphology.remove_small_objects(labels, min_size=min_area)
            np.savez_compressed(os.path.join(self.base_url, 'clean_segments.npz'), segmentation=labels)

        if False:
            print self.segmentation.shape, 'self.segmentation'
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            seq = self.segmentation[:10, :, :]
            for i in range(seq.shape[0]):
                a = fig.add_subplot(2, 5, i + 1)
                imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')

        if False:
            labels[labels > 0] = 1
            labels = labels[:10, :, :]
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            seq = labels[0:10, :, :]
            for i in range(seq.shape[0]):
                a = fig.add_subplot(2, 5, i + 1)
                imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')
        # plt.show()

    def find_ends(self, segmentation):
        # segmentation = segmentation[:10, :, :]
        ends_map = np.zeros_like(segmentation)
        unique_labels = np.unique(segmentation)
        print len(unique_labels), 'unique_labels'
        for a in unique_labels:
            print a
            if a > 0:
                l = deepcopy(segmentation)
                l[l != a] = 0
                l_sum = np.sum(l, axis=(1, 2))
                inds = np.where(l_sum > 0)[0]
                if len(inds) >= 2:
                    begin, end = inds[0], inds[-1]

                    begin_slice = l[begin, :, :]
                    begin_slice[begin_slice > 0] = 1  # begin: 1
                    ends_map[begin, :, :] += begin_slice

                    end_slice = l[end, :, :]
                    end_slice[end_slice > 0] = 2  # end: 2
                    ends_map[end, :, :] += end_slice

        np.savez_compressed(os.path.join(self.base_url, 'ends_segments.npz'), map=ends_map)

    def find_ends_fast(self, segmentation):
        ends_map = np.zeros_like(segmentation)
        for i in range(segmentation.shape[0] - 1):
            print i
            a = segmentation[i, :, :]
            b = segmentation[i + 1, :, :]
            unique_a = np.unique(a)
            unique_b = np.unique(b)
            ends = np.setdiff1d(unique_a, unique_b)
            begins = np.setdiff1d(unique_b, unique_a)

            begins_2d = np.zeros_like(a)
            for be in begins:
                begins_2d[b == be] = 1

            ends_2d = np.zeros_like(a)
            for en in ends:
                ends_2d[a == en] = 2

            ends_map[i, :, :] += ends_2d
            ends_map[i + 1, :, :] += begins_2d
        np.savez_compressed(os.path.join(self.base_url, 'ends_segments.npz'), map=ends_map)

    def show_ends(self):
        map = np.load(os.path.join(self.base_url, 'ends_segments.npz'))
        map = map['map']
        map = map[:10, :, :]

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        seq = map[0:10, :, :]
        for i in range(seq.shape[0]):
            a = fig.add_subplot(2, 5, i + 1)
            imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')
        plt.show()

    def analyze_predicts(self):
        self.predicts = self.predicts[:, :, :, 0]
        self.segmentation = self.segmentation[:, ::2, ::2]

        # self.gt_matrix, self.mitosis_labels = self.get_groundtruth(self.segmentation)
        # self.mitosis_info = self.match_gt_predicts(self.gt_matrix, self.predicts)

        self.predicts[self.predicts > 0] = 2.0
        labels, number = label(self.predicts)
        print number, 'predicts number'  # 1706 | 1877

        event_lengths = []
        mitosis_lengths = []
        mean_length, percentile = 0, 0
        mitosis_avgpxl_len = []

        # data = np.load(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'raw_sequence', 'raw_sequence.npz'))
        # raw_tensor = data['sequence'][:, ::4, ::4]

        for i in range(1, number):
            l = deepcopy(labels)
            l[l != i] = 0

            l_sum = np.sum(l, axis=(1, 2))
            l_sum[l_sum > 0] = 1
            length = np.sum(l_sum)
            event_lengths.append(length)

            if False:
                l2 = deepcopy(l)
                l2[l2 > 0] = 1.0

                for j in range(l2.shape[0]):
                    l2[j, :, :] = binary_dilation(l2[j, :, :], iterations=3)


                avg_pxl = np.sum(raw_tensor * l2) / np.sum(l2)

                if avg_pxl > 0.0:  # 0.4:
                    print avg_pxl
                    mitosis_lengths.append(length)
                    matches = self.gt_matrix * l2
                    match = np.amax(matches)
                else:
                    match = 0


                info = None
                if match > 0:
                    location = np.where(self.mitosis_info[:, 0] == match)
                    if len(location[0]) > 0:
                        info = self.mitosis_info[location[0][0], :]
                        # mitosis_lengths.append(length)
                    else:
                        # continue
                        pass
                else:
                    # continue
                    pass

                if info is not None:
                    mitosis_avgpxl_len.append([info[0], info[1], length])
                    # print mitosis_avgpxl_len[-1], 'mitosis_avgpxl_len ---', i

            if i % 10 == 1:
                if len(event_lengths) > 0:
                    print length, 'len', i, number, '-', np.mean(event_lengths), '-', np.std(event_lengths), '-', np.percentile(event_lengths, 50), '-', np.percentile(event_lengths, 75), 'events'
                if len(mitosis_lengths) > 0:
                    print np.mean(mitosis_lengths), '-', np.percentile(mitosis_lengths, 50), '-', np.percentile(mitosis_lengths, 75), 'mitosis -', len(mitosis_lengths), 'len'
        print np.mean(event_lengths), 'mean -', np.percentile(event_lengths, 50), 'percentile 50 -', np.percentile(event_lengths, 75), 'percentile 75 -', 'events'
        # print np.mean(mitosis_lengths), 'mean -', np.percentile(mitosis_lengths, 50), 'percentile 50 -', np.percentile(mitosis_lengths, 75), 'percentile 75 -', 'mitosis -', len(mitosis_lengths), 'len'

        np.savez_compressed('/media/newhd/Ha/my_env/cell_5_segmentation_F0005/mitosis_avgpxl_len.npz',
                            mitosis_avgpxl_len=mitosis_avgpxl_len, event_lengths=event_lengths, mitosis_lengths=mitosis_lengths)

        # print number, 'no'

        # self.segmentation = self.segmentation.astype(np.float32)
        # self.segmentation[self.segmentation > 0] = 1.0
        # print np.amin(self.segmentation), np.amax(self.segmentation), np.mean(self.segmentation)
        # self.segmentation += self.predicts.astype(self.segmentation.dtype)
        #
        # # self.segmentation[self.predicts == 0] = 0.0
        # print np.sum(self.segmentation), 'sum'

        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        seq = self.segmentation[100:110, :, :]
        for i in range(seq.shape[0]):
            a = fig.add_subplot(2, 5, i + 1)
            imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')
        plt.show()
        """

    def average_cell_size(self, segmentation):
        print segmentation.shape, 'segmentation'
        segmentation = segmentation[:, ::2, ::2]
        labels, number = label(segmentation)
        a = np.arange(1, number)
        np.random.shuffle(a)
        # a = a[:500]
        cell_sizes = []

        for i, v in enumerate(a):
            l = deepcopy(labels)
            l[l != v] = 0
            l[l > 0] = 1
            l_sum = np.sum(l, axis=(1, 2))
            l_length = deepcopy(l_sum)
            l_length[l_length > 0] = 1
            length = np.sum(l_length)
            if length > 10:
                size = np.sum(l_sum) / length
                cell_sizes.append(size)
                if len(cell_sizes) > 0:
                    print np.mean(cell_sizes), np.amax(cell_sizes), np.amin(cell_sizes), 'mean size', length, '-', number, '-', i
                    # Results: mean size - 24.09

    def match_gt_predicts(self, gt_matrix, predicts):
        # predicts = binary_dilation(predicts, iterations=5)
        for i in range(predicts.shape[0]):
            predicts[i, :, :] = binary_dilation(predicts[i, :, :], iterations=3)

        matches = gt_matrix * predicts
        uniques_matches = np.unique(matches)
        uniques_matches = [i for i in uniques_matches if i > 0]
        print len(uniques_matches), len(np.unique(gt_matrix))  # 359, 466 | 352, 466
        print np.unique(predicts)

        data = np.load(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'raw_sequence', 'raw_sequence.npz'))
        raw_tensor = data['sequence'][:, ::4, ::4]

        mitosis_info = []
        for i in range(len(uniques_matches)):
            label_id = int(uniques_matches[i]) - 1
            zxy = self.mitosis_labels[label_id, :]
            z = int(zxy[0]) - 1
            x = int(zxy[1])
            y = int(zxy[2])
            small_seq = raw_tensor[z-2:z+3, x-3:x+4, y-3:y+4]
            if small_seq.shape == (5, 7, 7):
                average_pixel = np.mean(small_seq)
                mitosis_info.append([int(uniques_matches[i]), average_pixel])
        mitosis_info = np.array(mitosis_info)
        # print mitosis_info
        print np.percentile(mitosis_info[:, 1], 25), np.percentile(mitosis_info[:, 1], 75), 'mitosis_info ---'
        print np.amin(mitosis_info[:, 1]), np.amax(mitosis_info[:, 1])
        print mitosis_info.shape  # (348, 2)  | (340, 2)
        return mitosis_info

    def match_gt_segmentation(self):
        self.segmentation = self.segmentation[:, ::2, ::2]
        self.predicts = np.squeeze(self.predicts)
        segmentation = self.predicts  # self.segmentation +
        gt_matrix, _ = self.get_groundtruth(segmentation)
        # for i in range(segmentation.shape[0]):
        #     segmentation[i, :, :] = binary_dilation(segmentation[i, :, :], iterations=3)

        segmentation = binary_dilation(segmentation, iterations=3).astype(np.float32)
        # for i in range(segmentation.shape[0]):
        #     segmentation[i, :, :] = binary_erosion(segmentation[i, :, :], iterations=3)

        # segmentation = segmentation.astype(np.float32)

        # segmentation = 1.0 - segmentation

        matches = gt_matrix * segmentation
        uniques_matches = np.unique(matches)
        uniques_matches = [i for i in uniques_matches if i > 0]
        print len(uniques_matches), len(np.unique(gt_matrix))  # 110, 466
        print float(len(uniques_matches)) / float(len(np.unique(gt_matrix)))  #
        print np.sum(segmentation), np.prod(segmentation.shape), np.sum(segmentation) / np.prod(segmentation.shape)

        data = np.load(os.path.join('/media/newhd/Ha/data/BAEC/F0005', 'raw_sequence', 'raw_sequence.npz'))
        raw_tensor = data['sequence'][:, ::4, ::4]

        # matches = binary_dilation(matches, iterations=5)

        # for i in range(matches.shape[0]):
        #     matches[i, :, :] = binary_dilation(matches[i, :, :], iterations=10)
        matches = matches.astype(np.float32)

        matches = segmentation
        matches *= raw_tensor
        np.savez_compressed('/media/newhd/Ha/my_env/cell_5_segmentation_F0005/outside_segmentation_mitosis.npz', matches=matches)

    def show_outside_segmentation_mitosis(self):
        data = np.load('F:\GoogleDrive\Work\PyCharm\cell_5_segmentation_F0005\outside_segmentation_mitosis.npz')
        data = data['matches']
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        seq = data[150:170, :, :]
        for i in range(seq.shape[0]):
            a = fig.add_subplot(4, 5, i + 1)
            imgplot = plt.imshow(seq[i, :, :], cmap='Greys_r')
        plt.show()

    def get_groundtruth(self, segmentation):
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

        def get_test_labels(gt_list, res_reduce=4):
            labels = gt_list
            for i in range(labels.shape[0]):
                labels[i, 0] = labels[i, 0]
                labels[i, 1] = int(np.round(labels[i, 1] / res_reduce))
                labels[i, 2] = int(np.round(labels[i, 2] / res_reduce))
            return labels

        labels = get_test_labels(gt_list)

        def create_gt_matrix(labels, segmentation):
            gt_matrix = np.zeros([segmentation.shape[0] + 50, segmentation.shape[1] + 50, segmentation.shape[2] + 50])
            # gt_matrix -= 1
            for i in range(labels.shape[0]):
                l = labels[i, :]
                l = [int(l[j]) for j in range(len(l))]
                z = l[0] + 25
                x = l[1] + 25
                y = l[2] + 25
                gt_matrix[z - 1, x, y] = float(i + 1)
            gt_matrix = gt_matrix[25:-25, 25:-25, 25:-25]
            return gt_matrix

        gt_matrix = create_gt_matrix(labels, segmentation)
        return gt_matrix, labels



# base_url = 'F:\Cell_data_public\Albany\\from_server'
base_url = '/media/newhd/Ha/my_env/cell_5_segmentation_F0005'
process = PostProcess(base_url=base_url, create_clean_segments=False, create_ends_segments=False,
                      analyze_predicts=True, average_cell_size=False)