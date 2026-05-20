import abc
from itertools import product
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from pymatching import Matching
from _pauli import wedge, wedge_dense


class Decoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def adjust_L(self, L):
        return

    @abc.abstractmethod
    def decode2chain(self, syndrome):
        return

    @abc.abstractmethod
    def decode2logical(self, syndrome):
        return

    @abc.abstractmethod
    def decode2probs(self, syndrome):
        return
    
    
class XZ_PymatchDecoder(Decoder):
    def __init__(self, code, evaluate_log_prob=None):
        self.code = code
        self._xSzC = Matching(code.x_stabilizers())
        self._zSxC = Matching(code.z_stabilizers())
        self.logicals_xz = code.logical_xz()
        self.evaluate_log_prob = evaluate_log_prob

    def adjust_L(self, L):
        self.code = type(self.code)(L=L)
        self._xSzC = Matching(self.code.x_stabilizers())
        self._zSxC = Matching(self.code.z_stabilizers())
        self.logicals_xz = self.code.logical_xz()

    def decode_rep(self, syndrome):
        z_stab, x_stab = np.hsplit(syndrome, 2)
        z_c = self._xSzC.decode(x_stab)
        x_c = self._zSxC.decode(z_stab)
        return np.hstack((x_c, z_c))

    def decode_batch(self, syndrome):
        rep = np.array([self.decode_rep(i) for i in syndrome], dtype=np.uint8)
        rep = sp.csr_matrix(rep, dtype=np.uint8)
        return rep

    def decode2chain(self, syndrome):
        rep = self.decode_batch(syndrome)
        return rep

    def new_decode_batch(self, syndrome):
        z_stab, x_stab = np.hsplit(syndrome, 2)
        z_c = self._xSzC.decode_batch(x_stab)
        x_c = self._zSxC.decode_batch(z_stab)
        return sp.csr_matrix(np.hstack((x_c, z_c)), dtype=np.uint8)

    def decode2logical(self, syndrome):
        rep = self.new_decode_batch(syndrome)
        logical = wedge(rep, self.logicals_xz)
        logical = np.array(logical.todense())
        return logical

    @staticmethod
    def label_from_bits(bits):
        power = 2 ** (np.array(range(4))[::-1])
        return np.inner(bits, power)

    def decode2label(self, syndrome):
        logical = self.decode2logical(syndrome)
        label = self.label_from_bits(logical)
        return label

    def decode2probs(self, syndrome):
        raise NotImplementedError


class BruteForceDecoder(Decoder):
    def __init__(self, code, evaluate_log_prob):
        self.L = code.L
        self.r_point = code.r_point
        self.K = code.n_k_d[1]
        self.evaluate_log_prob = evaluate_log_prob

        self.stabilizer = np.array(code.stabilizers().todense())
        self.i_stabilizer = np.array(code.i_stabilizers().todense())
        self.logicals_xz = np.array(code.logical_xz().todense())
        self.i_astabilizer = np.array(code.anti_stabilizers())

        self.selection_m = BruteForceDecoder.selection_matrix(2*(self.L**2-1))
        self.stab_q = np.mod(self.selection_m @ self.i_stabilizer, 2)
        self.wedge_l_perm = wedge_dense(self.logicals_xz, self.logicals_xz)
        self.coset_logic = BruteForceDecoder.selection_matrix(2 * self.K)

    def adjust_L(self, L):
        raise NotImplementedError

    @staticmethod
    def selection_matrix(K):
        bits = [0, 1]
        m = np.array([np.array(i) for i in product(bits, repeat=K)])
        return m

    def get_representater(self, i_syndrome):
        rep_q = i_syndrome @ self.i_astabilizer
        return rep_q

    def syndrome2i_syndrome(self, syndrome):
        i_syndrome = np.delete(syndrome, [self.r_point, self.L ** 2 + self.r_point], axis=1)
        return i_syndrome

    def coset_prob(self, rep):
        coset = np.mod(rep + self.stab_q, 2)
        coset_prob = np.exp(self.evaluate_log_prob(coset))
        return coset_prob

    def get_coset_prob(self, i_syndrome):
        rep = self.get_representater(i_syndrome)
        cosets_k = np.roll(self.coset_logic, 2, axis=1)
        probs_coset = np.zeros(2 ** 4)
        for i, k in enumerate(cosets_k):
            logic = np.mod(k @ self.logicals_xz, 2)
            rep_with_logic = np.mod(rep + logic, 2)
            probs_coset[i] = np.sum(self.coset_prob(rep_with_logic))
        return probs_coset.reshape(4, 2, 2)

    def decode_instance(self, i_syndrome):
        probs_coset = self.get_coset_prob(i_syndrome)
        ml_coset = np.argmax(probs_coset)
        logicals = BruteForceDecoder.selection_matrix(2*self.K)[ml_coset, :]
        return logicals

    def decode2chain(self, syndrome):
        raise NotImplementedError

    def decode2logical(self, syndrome, n_jobs=-1):
        i_syndrome = self.syndrome2i_syndrome(syndrome)
        logicals = np.array(Parallel(n_jobs=n_jobs)(delayed(self.decode_instance)(i) for i in i_syndrome))
        return logicals

    def decode2probs(self, syndrome, n_jobs=-1):
        i_syndrome = self.syndrome2i_syndrome(syndrome)
        log_probs = np.array(Parallel(n_jobs=n_jobs)(delayed(self.get_coset_prob)(i) for i in i_syndrome))
        return log_probs









