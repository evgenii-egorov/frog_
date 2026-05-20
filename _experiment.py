import abc

import numpy as np
import scipy.sparse as sp
from _pauli import wedge, wedge_dense
import torch

from sklearn.metrics import confusion_matrix


def compare_logicals_wedge(decode_s_l, wedge_noise_l):
    confuse_m = decode_s_l != wedge_noise_l
    stat_error_vec = confuse_m.sum(axis=0)
    any_error = np.sum(confuse_m.sum(axis=1) > 0)
    return any_error, stat_error_vec


def mean_var_error(decode_s_l, wedge_noise_l):
    n = decode_s_l.shape[0]
    any_error, _ = compare_logicals_wedge(decode_s_l, wedge_noise_l)
    p = any_error / n
    std = np.sqrt(p * (1.-p) / n)
    return p, std


class Loader(metaclass=abc.ABCMeta):
    def __init__(self, code, noise_model, b_size, device=None, dtype=None, basic_decode2logical=None):
        self.b_size = b_size
        self.batch_generator = BatchGenerator(code, noise_model)

        self.device = device
        self.dtype = dtype

        self.basic_decode2logical = basic_decode2logical

    def adjust_L_p(self, L, p, new_basic_decode2logical=None):
        self.basic_decode2logical = new_basic_decode2logical
        self.batch_generator.adjust_L_p(L, p)

    @staticmethod
    def label_from_bits(bits):
        power = 2 ** (np.array(range(4))[::-1])
        return np.inner(bits, power)

    def adjust_labels_with_basic(self, syndrome, logical):
        logical_basic_decoder = self.basic_decode2logical(syndrome)
        out = np.mod(logical + logical_basic_decoder, 2)
        return out

    def general_sampler(self, b_size):
        e, syndrome_b, logical_b = self.batch_generator.generate_batch(b_size)
        if self.basic_decode2logical is not None:
            logical_b = self.adjust_labels_with_basic(syndrome_b, logical_b)
        logical_b = self.label_from_bits(logical_b)
        return e, logical_b, syndrome_b

    @abc.abstractmethod
    def sample(self, b_size=None):
        pass

    def get_noise_channel_stats(self, sample_size=int(1e5)):
        logical_b, syndrome_b = self.sample(sample_size)
        p, d = torch.hsplit(syndrome_b, 2)
        mean_p, std_p = torch.mean(p), torch.std(p)
        mean_d, std_d = torch.mean(d), torch.std(d)
        mean_syndrome = torch.tensor([mean_p, mean_d]).reshape(1,2,1,1)
        std_syndrome = torch.tensor([std_p, std_d]).reshape(1,2,1,1)

        _, p_class = torch.unique(logical_b, return_counts=True)
        p_class = p_class / torch.sum(p_class)
        return mean_syndrome, std_syndrome, p_class

    def make_evaluation(self, decoder, b_size=2048, total_size=1024**2):
        R = int(total_size / b_size)
        self.batch_generator.noise_model.qmc_flag = False
        acc_array, std_array = np.zeros(R), np.zeros(R)
        for b in range(R):
            labels, syndrome_b = self.sample(b_size)
            prediction = decoder.decode2label(syndrome_b)

            if self.dtype is not None:
                labels = labels.data.cpu().numpy()
                prediction = prediction.data.cpu().numpy()

            confusion = confusion_matrix(labels, prediction, labels=range(16))
            acc_array[b] = np.sum(np.diag(confusion)) / b_size
            std_array[b] = np.sqrt(acc_array[b] * (1. - acc_array[b]) / b_size)

        acc = np.mean(acc_array)
        std = np.sqrt(acc * (1. - acc) / (b_size * R))

        return acc, std, acc_array, std_array


class BatchGenerator:
    def __init__(self, code, noise_model):
        self.code = code
        self.noise_model = noise_model
        self.to_dense = lambda x: np.array(x.todense(), dtype=np.uint8)
        self.coef_sub = lambda a, b: self.to_dense(wedge(a, b))

    def adjust_L_p(self, L, p):
        new_code = type(self.code)(L=L)
        new_noise_mode = type(self.noise_model)(2*L**2, p)
        self.__init__(new_code, new_noise_mode)

    def get_syndrome(self, pauli):
        noise = pauli[None, :] if len(pauli.shape) < 2 else pauli
        return wedge_dense(noise, self.to_dense(self.code.stabilizers()))

    def decompose(self, noise):
        logical_n = self.coef_sub(noise, self.code.logical_xz())
        syndrome_n = self.coef_sub(noise, self.code.stabilizers())
        return logical_n, syndrome_n

    def generate_batch(self, b_size):
        noise = self.noise_model.sample(b_size)
        noise = sp.csr_matrix(noise,  dtype=np.uint8)
        logical_b, syndrome_b = self.decompose(noise)
        return noise, syndrome_b, logical_b


class TorchLoader(Loader):
    def sample(self, b_size=None):
        if b_size is None:
            b_size = self.b_size

        e, logical_b, syndrome_b = self.general_sampler(b_size)

        logical_b = torch.tensor(data=logical_b, dtype=torch.long, device=self.device)
        syndrome_b = torch.tensor(data=syndrome_b, dtype=self.dtype, device=self.device)
        return logical_b, syndrome_b


class NumpyLoader(Loader):
    def sample(self, b_size=None):
        if b_size is None:
            b_size = self.b_size
        e, logical_b, syndrome_b = self.general_sampler(b_size)
        return logical_b, syndrome_b

