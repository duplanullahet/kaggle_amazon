from labels_stats import LabelsStatistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import numpy as np
import random
import pickle
import glob
import os


IM_WIDTH = 224
IM_HEIGHT = 224


class KaggleAmazonData(object):
    def __init__(self, training_data_dir, training_data_dir_jpeg, training_labels_file, testing_data_dir,
                 validation_percentage=5.0):
        self.training_data_dir = training_data_dir
        self.training_data_dir_jpeg = training_data_dir_jpeg
        self.testing_data_dir = testing_data_dir
        self.testing_files = [os.path.basename(file) for file in sorted(glob.glob(self.testing_data_dir+'/*.tif'))]  # os.listdir(self.testing_data_dir)

        self.stats_pickle_file = './data/stats.pickle'
        self.labels_stats = LabelsStatistics(data_file=training_labels_file, scratch=False)
        self.compute_data_stats()

        labelled_image_count = len(self.labels_stats.image_names)
        self.training_size = int(labelled_image_count * (100 - validation_percentage) / 100)
        self.validation_size = labelled_image_count - self.training_size
        self.testing_size = len(self.testing_files)

        np.random.seed(0)  # not sure
        permuted_index = np.array(range(labelled_image_count))  # undo
        # permuted_index = np.random.permutation(labelled_image_count)
        self.training_indexes = permuted_index[:self.training_size]
        self.validation_indexes = permuted_index[self.training_size:]

    def _random_crop_im_data(self, file_list):
        """
        Random x/y shift, and random left-right and up-down shift
        """
        im_rgb_data = np.empty((len(file_list), IM_WIDTH, IM_HEIGHT, 3))
        im_ir_data = np.empty((len(file_list), IM_WIDTH, IM_HEIGHT))
        index = 0
        for file in file_list:
            img = io.imread(file)
            # random flip left/right
            if random.random() <= 0.5:
                img = np.fliplr(img)
            # random flip up/down
            if random.random() <= 0.5:
                img = np.flipud(img)
            # random shift along x/y
            shift_w = random.randint(0, img.shape[0] - IM_WIDTH)
            shift_h = random.randint(0, img.shape[1] - IM_HEIGHT)
            im_rgb_data[index, :] = self._calibrate_image(img[shift_w:shift_w+IM_WIDTH, shift_h:shift_h+IM_HEIGHT, 0:3])
            im_ir_data[index, :] = self._scale_ir(img[shift_w:shift_w+IM_WIDTH, shift_h:shift_h+IM_HEIGHT, 3])
            index += 1
        return im_rgb_data, im_ir_data

    def _labelled_data_at_indexes(self, indexes):
        if type(indexes) is not list:
            indexes = [indexes]

        selected_images = self.labels_stats.image_names[indexes]
        selected_image_files = [os.path.join(self.training_data_dir, each+'.tif') for each in selected_images]

        rgb_data, ir_data = self._random_crop_im_data(selected_image_files)
        labels = self.labels_stats.labels_vecs[indexes]

        # print(labels.sum(axis=0).astype(np.int))
        # for p, c in zip(self.labels_stats.single_p, labels.sum(axis=0).astype(np.int)):
        #     print('{}\t{:.3f}'.format(c, p))

        return selected_images, rgb_data, ir_data, labels

    def testing_image(self, image_index):
        selected_image_files = [os.path.join(self.testing_data_dir, self.testing_files[image_index])]
        rgb_data, ir_data = self._random_crop_im_data(selected_image_files)
        image_names = self.testing_files[image_index][:-4]
        return image_names, rgb_data, ir_data, None

    def training_image(self, image_indexes):
        return self._labelled_data_at_indexes(self.training_indexes[image_indexes])

    def _sample_categories(self, sample_count):
        def __sample_from_p(p):
            r = random.random()
            i = -1
            sm = 0
            while sm < r and i < len(p):
                i += 1
                sm += p[i]
            return i

        category_count = len(self.labels_stats.labels_set)

        training_labels_vecs = self.labels_stats.labels_vecs[self.training_indexes, :]
        # training_image_names = self.labels_stats.image_names[self.training_indexes]

        positive = [np.where(training_labels_vecs[:, i] == 1)[0] for i in range(category_count)]
        negative = [np.where(training_labels_vecs[:, i] == 0)[0] for i in range(category_count)]

        sample_indexes = np.empty(sample_count, dtype=np.int)
        # hst = np.zeros(category_count, dtype=np.int)
        for sample_index in range(sample_count):
            category_index = __sample_from_p(p=np.ones(category_count) / float(category_count))

            # hst[category_index] += 1

            selected_list = positive if random.random() < (1-self.labels_stats.single_p[category_index]) else negative

            sample_indexes[sample_index] = random.choice(selected_list[category_index])

        # print(hst)
        # print(sample_indexes)
        # # print(self.labels_vecs[sample_indexes, :].sum(axis=0).astype(np.int))
        return sample_indexes

    def random_training_batch(self, batch_size):
        return self.training_image(image_indexes=self._sample_categories(sample_count=batch_size))  # undo
        # return self.training_image(image_indexes=np.random.permutation(self.training_size)[:batch_size])

    def validation_image(self, image_indexes):
        return self._labelled_data_at_indexes(self.validation_indexes[image_indexes])

    # def _scale_rgb(self, img):
    #     img = img.astype(np.float32)
    #     # print('rgb before scaling:', img.min(), img.max(), img.mean())
    #
    #     rgb_min = float(self.data_min[0:3].min())
    #     rgb_max = float(self.data_max[0:3].max())
    #
    #     data_mean = self.data_mean[0:3].copy()
    #     data_mean -= rgb_min
    #     data_mean /= rgb_max
    #     # print(data_mean)
    #
    #     img -= rgb_min
    #     img /= rgb_max
    #     # for channel in range(3):
    #     #     img[:, :, channel] -= data_mean[channel]
    #
    #     # print('rgb after scaling:', img.min(), img.max(), img.mean())
    #     return img

    def _scale_ir(self, img):
        img = img.astype(np.float32)
        # print('ir before scaling:', img.min(), img.max(), img.mean())

        ir_min = float(self.data_min[3].min())
        ir_max = float(self.data_max[3].max())

        data_mean = self.data_mean[3].copy()
        data_mean -= ir_min
        data_mean /= (ir_max / 255.)
        # print(data_mean)

        img -= ir_min
        img /= (ir_max / 255.)
        # img -= data_mean

        # print('ir after scaling:', img.min(), img.max(), img.mean())
        return img

    def _load_stats_file(self):
        try:
            with open(self.stats_pickle_file, 'rb') as f:
                self.data_min, self.data_max, self.data_mean, self.ref_means, self.ref_stds = pickle.load(f)
                return True
        except:
            return False

    def _save_stats_to_file(self):
        var_list = [self.data_min, self.data_max, self.data_mean, self.ref_means, self.ref_stds]

        with open(self.stats_pickle_file, 'wb') as f:
            pickle.dump(var_list, f)

    def compute_data_stats(self):
        if self._load_stats_file():
            return

        self._compute_jpeg_hist()
        self._compute_min_max_mean()
        self._save_stats_to_file()

    def _compute_jpeg_hist(self, sample_count=2000):
        """
        Computes histogram for all jpeg training data files.
        """
        ref_colors = [[], [], []]
        all_jpegs = glob.glob(self.training_data_dir_jpeg+'/*.jpg')
        selected_jpegs = [all_jpegs[i] for i in np.random.permutation(len(all_jpegs))[:sample_count].tolist()]
        for counter, file in enumerate(selected_jpegs):
            print(counter, file)

            img = mpimg.imread(file)[:, :, :3]
            # Flatten 2-D to 1-D
            data = img.reshape((-1, 3))

            # Dump pixel values to aggregation buckets -- badly suboptimal
            for i in range(3):
                ref_colors[i] = ref_colors[i] + data[:, i].tolist()

        ref_colors = np.array(ref_colors)
        self.ref_means = [np.mean(ref_colors[i]) for i in range(3)]
        self.ref_stds = [np.std(ref_colors[i]) for i in range(3)]

    def _compute_min_max_mean(self):
        """
        Calculates self.data_min, self.data_max, self.data_mean for the training data.
        """
        self.data_min = np.array([9999999.0, 9999999.0, 9999999.0, 9999999.0])
        self.data_max = np.array([0.0, 0.0, 0.0, 0.0])
        self.data_mean = np.array([0.0, 0.0, 0.0, 0.0])

        all_tiffs = glob.glob(self.training_data_dir+'/*.tif')
        for counter, file in enumerate(all_tiffs):
            if counter % 50 == 0:
                print(counter, file)

            img = io.imread(os.path.join(self.training_data_dir, file))

            for channel in range(4):
                _min = img[:, :, channel].min()
                _max = img[:, :, channel].max()
                self.data_mean[channel] += img[:, :, channel].mean() / float(len(all_tiffs))

                if _min < self.data_min[channel]:
                    self.data_min[channel] = _min
                if self.data_max[channel] < _max:
                    self.data_max[channel] = _max

    def _debug_image_channels(self, tif_img, jpg_img=None):
        b, g, r, nir, rgb = tif_img[:, :, 0], tif_img[:, :, 1], tif_img[:, :, 2], tif_img[:, :, 3],\
                            self._calibrate_image(tif_img[:, :, :3])
        fig = plt.figure(1, figsize=(16, 4))
        plot_count = 5 if jpg_img is None else 6
        for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'), (rgb, 'rgb'))):
            a = fig.add_subplot(1, plot_count, i + 1)
            a.set_title(c)
            plt.imshow(x)
        if jpg_img is not None:
            a = fig.add_subplot(1, plot_count, plot_count)
            a.set_title('jpeg')
            plt.imshow(jpg_img)
        plt.show()

    def _calibrate_image(self, rgb_image):
        # Transform test image to 32-bit floats to avoid surprises when doing arithmetic with it
        calibrated_img = rgb_image.copy().astype('float32')

        # Loop over RGB
        for i in range(3):
            # Subtract mean
            calibrated_img[:, :, i] = calibrated_img[:, :, i] - np.mean(calibrated_img[:, :, i])
            # Normalize variance
            calibrated_img[:, :, i] = calibrated_img[:, :, i] / np.std(calibrated_img[:, :, i])
            # Scale to reference
            calibrated_img[:, :, i] = calibrated_img[:, :, i] * self.ref_stds[i] + self.ref_means[i]
            # Clip any values going out of the valid range
            calibrated_img[:, :, i] = np.clip(calibrated_img[:, :, i], 0, 255) / 255.

        # Convert to 8-bit unsigned int
        # return calibrated_img.astype('uint8')
        return calibrated_img

    def show_train_images(self):
        all_tiffs = sorted(glob.glob(self.training_data_dir+'/*.tif'))
        all_jpegs = sorted(glob.glob(self.training_data_dir_jpeg+'/*.jpg'))
        for files in zip(all_tiffs, all_jpegs):
            tif_img = io.imread(files[0])
            jpg_img = mpimg.imread(files[1])[:, :, :3]
            self._debug_image_channels(tif_img, jpg_img)
            print(files)


if __name__ == '__main__':
    training_data_root = os.path.join(os.path.expanduser("~"), 'Developer/data/kaggle_amazon/')
    data_source = KaggleAmazonData(
        training_data_dir=os.path.join(training_data_root, 'train-tif-v2'),
        training_data_dir_jpeg=os.path.join(training_data_root, 'train-jpg'),
        training_labels_file=os.path.join(training_data_root, 'train_v2.csv'),
        testing_data_dir=os.path.join(training_data_root, 'test-tif-v2'))
    im_names, rgb_data, ir_data, labels = data_source.random_training_batch(batch_size=60)
    # data_source.show_train_images()
