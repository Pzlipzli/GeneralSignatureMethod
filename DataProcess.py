import numpy as np
import warnings
import iisignature
import math
"""
Data augmentation includes three main methods:
x = (x1, x2, ..., xn), t belongs to {0, 1, 2, ..., n}, xi is d dimensional vector
1. Sensitivity Introduction
    a Time augmentation: x -> ((t1, x1), (t2, x2), ..., (tn, xn))
    b Basepoint augmentation: x -> (0, x1, x2, ..., xn)
2. Lowering Dimensionality
    a ~ pairs: x -> ((t, x_{1}, x_{2}), (t, x_{1}, x_{3}), ..., (t, x_{d-1}, x_{d}))
    b ~ triplets: similarly defined
    c ~ random: x -> (A1(t, x), A2(t, x), ..., Ap(t, x)), where Ai: d -> e
3. Information Extraction
    a Lead-lag transformation: x -> ((x1, x1), (x2, x1), (x2, x2), ..., (xn, xn))
        For lead: xtj = xti if j = 2i or 2i - 1
        For lag : xtj = xti if j = 2i or 2i + 1
"""


class SignatureAugment:
    def __init__(self):
        pass

    def time_augmentation(self, df_data):
        """
        Data augmentation, adding a time dimension
        x -> ((t1, x1), (t2, x2), ..., (tn, xn))
        """
        df_data.index += 1
        array = np.array(df_data.reset_index())

        return array

    def basepoint_augmentation(self, array):
        """
        Basepoint augmentation, adding a basepoint
        x -> (0, x1, x2, ..., xn)
        """
        zero_row = np.zeros((1, array.shape[1]))
        new_array = np.insert(array, 0, zero_row, axis=0)

        return new_array

    def pair_projection(self, array):
        """
        x -> ((t, x_{1}, x_{2}), (t, x_{1}, x_{3}), ..., (t, x_{d-1}, x_{d}))
        """
        arrays = []
        for i in range(1, array.shape[1]):
            for j in range(1, array.shape[1]):
                if j != i:
                    arrays.append(array[:, [0, i, j]])
        return arrays

    def triple_projection(self, array):
        """
        x -> ((t, x_{1}, x_{2}, x_{3}), (t, x_{1}, x_{2}, x_{4}), ..., (t, x_{d-2}, x_{d-1}, x_{d}))
        """
        arrays = []
        for i in range(1, array.shape[1]):
            for j in range(1, array.shape[1]):
                for k in range(1, array.shape[1]):
                    if j != i and i != k and j != k:
                        arrays.append(array[:, [0, i, j, k]])
        return arrays

    def random_projection(self, array, size):
        """
        x -> (A1(t, x), A2(t, x), ..., Ap(t, x))
        A: d -> e
        """
        arrays = []
        e, p = size
        d = array.shape[1]

        random_matrix = np.random.rand(p, d, e)
        for i in range(p):
            arrays.append(np.matmul(array, random_matrix[i]))

        return arrays

    def enlarge_windows(self, arrays, n):
        """
        Enlarge windows, adding a basepoint after enlarging the window
        """
        data_list = []
        for array in arrays:
            data_list.extend([self.basepoint_augmentation(array[:array.shape[0] // n * i]) for i in range(1, n + 1)])

        return data_list

    def all_windows(self, arrays):
        """
        Whole window, only adding a basepoint
        """
        data_list = []
        for array in arrays:
            data_win = self.basepoint_augmentation(array)
            data_list.append(data_win)

        return data_list

    def sliding_windows(self, arrays, slides: tuple):
        """
        Sliding window, adding a basepoint after sliding
        """
        length, stride = slides
        data_list = []
        for array in arrays:
            data_list.extend([self.basepoint_augmentation(array[i:i + length])
                              for i in range(0, array.shape[0] - length + 1, stride)])

        return data_list

    def hierarchical_dyadic_windows(self, arrays, order):
        """
        Hierarchical dyadic window, adding a basepoint after enlarging the window
        """
        data_list = []
        for array in arrays:
            length = array.shape[0]
            data_list.extend([self.basepoint_augmentation(array[i:i + length // (2 ** j)])
                              for j in range(1, order + 1)
                              for i in range(0, array.shape[0] - array.shape[0] // (2 ** j) + 1, array.shape[0] // (2 ** j))])
        return data_list


    def lead_lag(self, data_list):
        """
        Turn the series into lead-lag series
        """
        data_lists = []
        for data in data_list:
            data_lead_lag = np.repeat(data, repeats=2, axis=0)
            lead_lag_list = []
            lead_lag_list.append(data_lead_lag[1:])  # lead series
            lead_lag_list.append(data_lead_lag[:-1])  # lag series

            data_lists.append(np.hstack(lead_lag_list))

        return data_lists

    def run_augment(self, data_token, sig_depth, projection='pair', size: tuple = None, window='all', win_num=None,
                    slides=None, order=None, lead_lag=False, log=False, scale=None):

        data_after_time_aug = self.time_augmentation(data_token)

        if projection == 'pair':
            data_after_dimension_reduction = self.pair_projection(data_after_time_aug)
        elif projection == 'triple':
            data_after_dimension_reduction = self.triple_projection(data_after_time_aug)
        elif projection == 'random':
            if not size:
                raise ValueError("size is not set")
            data_after_dimension_reduction = self.random_projection(data_after_time_aug, size)
        else:
            raise ValueError("Invalid projection method")

        if window == 'enlarge':
            if not win_num:
                warnings.warn("win_num is not set, use default value 3")
                data_after_windows = self.enlarge_windows(data_after_dimension_reduction, n=3)
            else:
                data_after_windows = self.enlarge_windows(data_after_dimension_reduction, n=win_num)
        elif window == 'all':
            data_after_windows = self.all_windows(data_after_dimension_reduction)
        elif window == 'slide':
            if not slides:
                raise ValueError("slides is not set")
            data_after_windows = self.sliding_windows(data_after_dimension_reduction, slides=slides)
        elif window == 'dyadic':
            if not order:
                raise ValueError("order is not set")
            data_after_windows = self.hierarchical_dyadic_windows(data_after_dimension_reduction, order=order)
        else:
            raise ValueError("Invalid window method")

        if lead_lag:
            data_after_lead_lag = self.lead_lag(data_after_windows)
        else:
            data_after_lead_lag = data_after_windows

        if scale == 'pre':
            data_after_scale = data_after_lead_lag * (math.factorial(sig_depth) ** (1 / sig_depth))
        else:
            data_after_scale = data_after_lead_lag

        if log:
            sig_list = []
            for array in data_after_scale:
                d = array.shape[1]
                prep = iisignature.prepare(d, sig_depth, "O")
                ssig = iisignature.logsig(array, prep)
                if scale == 'post':
                    sig_list.append(post_signature_scaling(ssig, array, sig_depth))
                else:
                    sig_list.append(ssig)
        else:
            sig_list = []
            for array in data_after_scale:
                ssig = iisignature.sig(array, sig_depth)
                if scale == 'post':
                    sig_list.append(post_signature_scaling(ssig, array, sig_depth))
                else:
                    sig_list.append(ssig)

        concatenated_sig = np.concatenate(sig_list)

        return concatenated_sig


def post_signature_scaling(ssig, array, sig_depth):
    """
    scaling function
    """
    for k in range(1, sig_depth + 1):
        start_idx = calculate_sigsize(array.shape[1], k - 1)
        end_idx = calculate_sigsize(array.shape[1], k)

        ssig[start_idx:end_idx] *= math.factorial(k)
    return ssig


def calculate_sigsize(d, m):
    """
    calculate the size of the signature
    """
    size = 0
    for k in range(1, m + 1):
        size += d ** k
    return size
