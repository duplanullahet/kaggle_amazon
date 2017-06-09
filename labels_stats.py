import numpy as np
import pickle
import random
import csv
import os


class LabelsStatistics(object):
    """
    Calculates labels statistics, such as single and pairwise probabilities of labels.
    Works on labels provided for https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
    """

    def __init__(self, data_file=None, scratch=False):
        self.pickle_file = './data/labels_stats.pickle'

        if not self.load_data_from_file() or scratch:
            self.image_names, self.list_of_list_of_image_labels = self.load_labels(data_file)
            self.labels_set = self.unique_labels(self.list_of_list_of_image_labels)
            self.labels_dict = self.labels_set_to_dictionary(self.labels_set)
            self.labels_dict_inv = dict((v, k) for k, v in self.labels_dict.items())
            self.labels_vecs, self.labels_vecs_for_names_dict = self.labels_vecs_for_names(self.image_names, self.list_of_list_of_image_labels, self.labels_dict)
            self.sample_count = len(self.list_of_list_of_image_labels)
            self.single_counts, self.pairwise_counts, self.positive_pairwise_counts = \
                self.count_labels(self.list_of_list_of_image_labels, self.labels_dict)
            self.single_p, self.pairwise_p = self.labels_probs(self.single_counts, self.pairwise_counts, self.sample_count)
            self.save_data_to_file()

    def labels_vecs_for_names(self, image_names, list_of_list_of_image_labels, labels_dict):
        labels_vecs = []
        labels_vecs_for_names_dict = dict()
        for i in range(len(image_names)):
            image_name = image_names[i]
            labels = list_of_list_of_image_labels[i]
            labels_vec = np.zeros(len(labels_dict))
            for label in labels:
                labels_vec[labels_dict[label]] = 1.0
            
            labels_vecs.append(labels_vec)
            labels_vecs_for_names_dict.update({image_name: labels_vec})

        return np.array(labels_vecs), labels_vecs_for_names_dict

    def load_data_from_file(self):
        try:
            with open(self.pickle_file, 'rb') as f:
                self.image_names, self.list_of_list_of_image_labels, \
                self.labels_set, self.labels_dict, self.labels_dict_inv, \
                self.labels_vecs, self.labels_vecs_for_names_dict, \
                self.sample_count, self.single_counts, self.pairwise_counts, self.positive_pairwise_counts, \
                self.single_p, self.pairwise_p = pickle.load(f)
                return True
        except:
            return False

    def save_data_to_file(self):
        var_list = [self.image_names, self.list_of_list_of_image_labels,
                    self.labels_set, self.labels_dict, self.labels_dict_inv,
                    self.labels_vecs, self.labels_vecs_for_names_dict,
                    self.sample_count, self.single_counts, self.pairwise_counts, self.positive_pairwise_counts,
                    self.single_p, self.pairwise_p]
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(var_list, f)

    def debug(self):
        for i in range(len(self.single_p)):
            print('{}: {}\tp = {:.4f}'.format(self.labels_dict_inv[i].ljust(18, ' '), i, self.single_p[i]))

        index_2_pair, pair_2_index = self.pair_indices(len(self.labels_dict))
        np.set_printoptions(precision=4, suppress=True)
        for i in range(len(self.pairwise_p)):
            print('p({}, {}) = p({}, {}) ='.format(index_2_pair[i][0], index_2_pair[i][1],
                                                   self.labels_dict_inv[index_2_pair[i][0]],
                                                   self.labels_dict_inv[index_2_pair[i][1]]))
            print(self.pairwise_p[i])
            np.testing.assert_almost_equal(1.0, self.pairwise_p[i].sum())

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.imshow(positive_pp)
        # plt.figure(2)
        # plt.imshow(np.outer(p, p))
        # plt.figure(3)
        # plt.imshow(positive_pp - np.outer(p, p))
        # plt.show()

    def load_labels(self, data_file):
        with open(data_file) as f:
            reader = csv.reader(f, delimiter=',')
            data = np.array([each for each in reader])
            # data is an np.ndarray of train_i index and str (consisting the labels)
            # data = [['image_name', 'tags'],
            #         ['train_1', 'agriculture clear primary slash_burn water'],
            #         ['train_2', 'clear primary water'], ... ]

            # remove image_name column (column 0) and header row (row 0)
            data = data[1:]
            image_names = data[:, 0]
            image_labels = data[:, 1]

            # create list of list of tags/labels:
            list_of_list_of_image_labels = list(map(lambda x: x.split(' '), image_labels))
            # list_of_list_of_image_labels = [['agriculture', 'clear', 'primary', 'slash_burn', 'water'],
            #                           ['clear', 'primary', 'water'], ... ]

            return image_names, list_of_list_of_image_labels

    def unique_labels(self, list_of_list_of_image_labels):
        """
        :param list_of_list_of_image_labels: list of list of labels
        :return: set of unique labels
        """
        labels_set = set()
        for list_of_labels in list_of_list_of_image_labels:
            labels_set.update(set(list_of_labels))
        return labels_set

    def labels_set_to_dictionary(self, labels_set):
        """
        Assign unique value to each label in labels set.
        
        :param labels_set: set of labels
        :return: labels dict
        """
        label_value = 0
        labels_dict = {}
        for label in labels_set:
            labels_dict.update({label: label_value})
            label_value += 1
        return labels_dict

    def pair_indices(self, n):
        """
        Generate the following maps:
            pair_to_index[i, j] -> index
            index_to_pair[index] -> (i, j)
            where 0 <= i, j < n
        """
        counter = 0
        index_to_pair = []
        pair_to_index = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                index_to_pair.append((i, j))
                pair_to_index[i, j] = counter
                pair_to_index[j, i] = counter
                counter += 1
        return index_to_pair, pair_to_index

    def update_pairwise_counts(self, pairwise_counts, single_counts, pair_2_index):
        """
        Updated pairwise_counts based on the current single_counts.
        pairwise_counts stores the number of co-occurrence for each (i, j) pairs
        """
        n = len(single_counts)
        for i in range(n):
            for j in range(i + 1, n):
                c_i = np.array([single_counts[i], 1 - single_counts[i]])
                c_j = np.array([single_counts[j], 1 - single_counts[j]])
                c_ij = np.outer(c_i, c_j)
                pairwise_counts[pair_2_index[i, j]] += c_ij
        return pairwise_counts

    def count_labels(self, list_of_list_of_image_labels, labels_dict):
        single_counts = np.zeros(len(labels_dict))
        pairwise_counts = []
        positive_pairwise_counts = np.zeros((len(labels_dict), len(labels_dict)))

        index_2_pair, pair_2_index = self.pair_indices(len(labels_dict))
        for _ in range(len(index_2_pair)):
            pairwise_counts.append(np.zeros((2, 2)))

        for list_of_labels in list_of_list_of_image_labels:
            single_counts_tmp = np.zeros(len(labels_dict))
            for label in list_of_labels:
                single_counts_tmp[labels_dict[label]] += 1

            single_counts += single_counts_tmp
            pairwise_counts = self.update_pairwise_counts(pairwise_counts, single_counts_tmp, pair_2_index)
            positive_pairwise_counts += np.outer(single_counts_tmp, single_counts_tmp)

        return single_counts, pairwise_counts, positive_pairwise_counts

    def labels_probs(self, single_counts, pairwise_counts, sample_count):
        single_probs = single_counts / float(sample_count)
        pairwise_probs = []
        for i in range(len(pairwise_counts)):
            pairwise_probs.append(pairwise_counts[i] / float(sample_count))
        return single_probs, pairwise_probs

    def _sample_from_p(self, p):
        r = random.random()
        i = -1
        sm = 0
        while sm < r and i < len(p):
            i += 1
            sm += p[i]
        return i

    def sample_categories(self, sample_count, uniform_scale=1.4):
        """
        uniform_scale is a tricky number, which determines the uniformity of samples.
        See equation below for category_probabilities.
        """
        category_count = len(self.labels_set)

        # # category_probabilities = np.log(self.single_p + uniform_scale) / np.log(self.single_p + uniform_scale).sum()
        # category_probabilities = 1.0 - self.single_p
        # category_probabilities /= category_probabilities.sum()
        # print(category_probabilities)

        positive = [np.where(self.labels_vecs[:, i] == 1)[0] for i in range(category_count)]
        negative = [np.where(self.labels_vecs[:, i] == 0)[0] for i in range(category_count)]

        sample_indexes = np.empty(sample_count, dtype=np.int)
        hst = np.zeros(category_count, dtype=np.int)
        for sample_index in range(sample_count):
            # category_index = self._sample_from_p(category_probabilities)
            category_index = self._sample_from_p(np.ones(category_count) / float(category_count))

            hst[category_index] += 1

            # selected_list = positive if random.random() < 0.5 else negative
            selected_list = positive if random.random() < (1-self.single_p[category_index]) else negative

            sample_indexes[sample_index] = random.choice(selected_list[category_index])

        print(hst)
        # print(self.labels_vecs[sample_indexes, :].sum(axis=0).astype(np.int))
        return sample_indexes


if __name__ == '__main__':
    labels_stats = LabelsStatistics(
        data_file=os.path.join(os.path.expanduser("~"), 'Developer/data/kaggle_amazon', 'train_v2.csv'),
        scratch=False)
    labels_stats.debug()
    # labels_stats.sample_categories(17000, uniform_scale=10)
