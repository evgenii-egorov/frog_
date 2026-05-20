import abc
from functools import lru_cache
from _pauli import *
import scipy.sparse as sp
import galois
import numpy as np


class StabilizerCode(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def stabilizers(self):
        return

    @property
    @abc.abstractmethod
    def logical_x(self):
        return

    @property
    @abc.abstractmethod
    def logical_z(self):
        return

    @property
    @abc.abstractmethod
    def logical_xz(self):
        return

    @property
    @abc.abstractmethod
    def anti_stabilizers(self):
        return


class ToricCode(StabilizerCode):
    def __init__(self, L, r_point=0):
        self.r_point = r_point
        self.n_k_d = 2 * L ** 2, 2, L
        self.L = L
        self.N = 2 * (L ** 2)  # L ** 2 vertical, L ** 2 horizontal
        self.rxS = L ** 2 - 1  # vertex
        self.rzS = L ** 2 - 1  # plaquette
        self.rL = 2*self.N - 2 * (self.rxS + self.rzS)  # logical are rest

    @staticmethod
    def get_permutations(L):
        diags = [np.ones(L), np.ones(L-1), [1]]
        gen = sp.diags(diags, offsets=[0, 1, -L+1], dtype=np.uint8, format="csr")
        return gen

    @lru_cache(maxsize=None)
    def z_stabilizers(self):
        gen = ToricCode.get_permutations(self.L)
        eye = sp.eye(self.L, dtype=np.uint8, format="csr")
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        z_stab = sp.bmat([[non_zero_h, non_zero_v]], format="csr", dtype=np.uint8)
        return z_stab

    @lru_cache(maxsize=None)
    def x_stabilizers(self):
        gen = ToricCode.get_permutations(self.L).T
        eye = sp.eye(self.L, dtype=np.uint8, format="csr")
        non_zero_h = sp.kron(eye, gen, "csr")
        non_zero_v = sp.kron(gen, eye, "csr")
        x_stab = sp.bmat([[non_zero_h, non_zero_v]], format="csr", dtype=np.uint8)
        return x_stab

    @lru_cache(maxsize=None)
    def stabilizers(self):
        gen = ToricCode.get_permutations(self.L)
        eye = sp.eye(self.L, dtype=np.uint8, format="csr")
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        all_stab = sp.bmat([
            [None, None, non_zero_h, non_zero_v],
            [non_zero_v.T, non_zero_h.T, None, None]
        ], format="csr", dtype=np.uint8)
        return all_stab

    @lru_cache(maxsize=None)
    def i_stabilizers(self):
        mask = np.ones(self.L ** 2, dtype=bool)
        mask[self.r_point] = False
        all_stab = self.stabilizers()
        x, z = all_stab[:self.rxS+1], all_stab[self.rxS+1:]
        x = x[mask]
        z = z[mask]
        i_stab = sp.vstack((x, z), format="csr", dtype=np.uint8)
        return i_stab

    def ref_point_stabilizers(self):
        mask = np.zeros(self.L ** 2, dtype=bool)
        mask[self.r_point] = True
        all_stab = self.stabilizers()
        x, z = all_stab[:self.rxS + 1], all_stab[self.rxS + 1:]
        x = x[mask]
        z = z[mask]
        ref_stab = sp.vstack((x, z), format="csr", dtype=np.uint8)
        return ref_stab

    @lru_cache(maxsize=None)
    def i_normilizer(self):
        i_stab = self.i_stabilizers()
        logical = self.logical_xz()
        return sp.vstack((i_stab, logical), format="csr", dtype=np.uint8)

    @lru_cache(maxsize=None)
    def logical_x(self):
        gen = sp.csr_matrix(np.ones((1, self.L), dtype=np.uint8))
        eye = sp.csr_matrix(([1], ([0], [0])), shape=(1, self.L), dtype=np.uint8)
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        zero = sp.csr_matrix(non_zero_v.shape, dtype=np.uint8)
        x_logicals = sp.bmat([
            [non_zero_h, zero, zero, zero],
            [zero, non_zero_v, zero, zero]
        ], format="csr", dtype=np.uint8)
        return x_logicals

    @lru_cache(maxsize=None)
    def logical_z(self):
        gen = sp.csr_matrix(np.ones((1, self.L), dtype=np.uint8))
        eye = sp.csr_matrix(([1], ([0], [0])), shape=(1, self.L), dtype=np.uint8)
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        zero = sp.csr_matrix(non_zero_v.shape, dtype=np.uint8)
        z_logicals = sp.bmat([
            [zero, zero, non_zero_v, zero],
            [zero, zero, zero, non_zero_h]
        ], format="csr", dtype=np.uint8)
        return z_logicals

    @lru_cache(maxsize=None)
    def logical_xz(self):
        x_logical = self.logical_x()
        z_logical = self.logical_z()
        all_logical = sp.vstack([x_logical,
                                 z_logical])

        return all_logical

    @lru_cache(maxsize=None)
    def anti_stabilizers(self):  # general dense
        def get_sub_r(s_r):
            col = [0]
            for r in range(1, s_r.shape[0]):
                col.append(np.where(s_r[r] == 1)[0][0])
            return col

        def solve_reduced(s_r, r_p):
            columns = get_sub_r(s_r)
            full_rank = s_r[:, columns]
            full_rank_inv = np.linalg.inv(full_rank)
            sub_solution = full_rank_inv @ r_p
            solution = np.zeros((s_r.shape[1], r_p.shape[1]), dtype=np.uint8)
            solution[columns, :] = sub_solution
            return solution.T

        i_normilizer = self.i_normilizer()
        r_SL = i_normilizer.shape[0]
        r_S = self.rxS + self.rzS
        GF2 = galois.GF(2)
        I = sp.csr_matrix(([1]*r_S, (range(r_S), range(r_S))), shape=(r_SL, r_S), dtype=np.uint8)
        i2r = sp.hstack([i_normilizer, I], format="csr", dtype=np.uint8)
        i2r = GF2(i2r.todense()).row_reduce()

        s_p = i2r[:, :i_normilizer.shape[1]]
        r_p = i2r[:, i_normilizer.shape[1]:]
        aS = solve_reduced(s_p, r_p)
        aS = np.roll(aS, aS.shape[1]//2, axis=1)

        gramm = wedge_dense(aS, aS)
        gramm_l = np.tril(gramm)

        i_stab = self.i_stabilizers()
        aS = np.mod(aS + gramm_l.T @ i_stab.todense(), 2)

        return np.array(aS, dtype=np.uint8)


class ToricCode_d(ToricCode):
    def __init__(self, L, d, r_point=0):
        self.r_point = r_point
        self.n_k_d = 2 * L ** 2, 2, L
        self.L = L
        self.N = 2 * (L ** 2)  # L ** 2 vertical, L ** 2 horizontal
        self.rxS = L ** 2 - 1  # vertex
        self.rzS = L ** 2 - 1  # plaquette
        self.rL = 2*self.N - 2 * (self.rxS + self.rzS)  # logical are rest 
        self.d = d
        self.inv = np.mod(-1,d) 
    
    @staticmethod
    def get_permutations(L,d):
        inv = np.mod(-1,d)
        diags = [np.ones(L), np.full(L-1,inv), [inv]]
        gen = sp.diags(diags, offsets=[0, 1, -L+1], dtype=np.uint8,format="csr")
        return gen

    @lru_cache(maxsize=None)
    def z_stabilizers(self):
        gen = ToricCode_d.get_permutations(self.L,self.d)
        eye = sp.eye(self.L, dtype=np.uint8, format="csr")
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        z_stab = sp.bmat([[non_zero_h, non_zero_v]], format="csr", dtype=np.uint8)
        return z_stab

    @lru_cache(maxsize=None)
    def x_stabilizers(self):
        gen = ToricCode_d.get_permutations(self.L,self.d).T
        eye = sp.eye(self.L, dtype=np.uint8, format="csr")
        gen_r = gen.multiply(-1)
        gen_r.data %= self.d
        non_zero_h = sp.kron(eye, gen, "csr")
        non_zero_v = sp.kron(gen_r, eye, "csr")
        x_stab = sp.bmat([[non_zero_h, non_zero_v]], format="csr", dtype=np.uint8)
        return x_stab

    @lru_cache(maxsize=None)
    def stabilizers(self):
        gen = ToricCode_d.get_permutations(self.L,self.d)
        eye = sp.eye(self.L, dtype=np.int8, format="csr")
        non_zero_h = sp.kron(gen, eye, "csr")
        non_zero_v = sp.kron(eye, gen, "csr")
        non_zero_h_r = non_zero_h.multiply(-1)
        non_zero_h_r.data %= self.d
        all_stab = sp.bmat([
            [None, None, non_zero_h, non_zero_v],  
            [non_zero_v.T, non_zero_h_r.T, None, None]
        ], format="csr", dtype=np.int8)
        return all_stab 
 
    @lru_cache(maxsize=None)
    def anti_stabilizers(self):  # general dense
        def get_sub_r(s_r):
            col = [0]
            for r in range(1, s_r.shape[0]):
                col.append(np.where(s_r[r] == 1)[0][0])
            return col

        def solve_reduced(s_r, r_p):
            columns = get_sub_r(s_r)
            full_rank = s_r[:, columns]
            full_rank_inv = np.linalg.inv(full_rank)
            sub_solution = full_rank_inv @ r_p
            solution = np.zeros((s_r.shape[1], r_p.shape[1]), dtype=np.uint8)
            solution[columns, :] = sub_solution
            return solution.T

        i_normilizer = self.i_normilizer()
        r_SL = i_normilizer.shape[0]
        r_S = self.rxS + self.rzS
        GF_d = galois.GF(self.d)
        I = sp.csr_matrix(([1]*r_S, (range(r_S), range(r_S))), shape=(r_SL, r_S), dtype=np.uint8)
        i2r = sp.hstack([i_normilizer, I], format="csr", dtype=np.uint8)
        i2r = GF_d(i2r.todense()).row_reduce()

        s_p = i2r[:, :i_normilizer.shape[1]]
        r_p = i2r[:, i_normilizer.shape[1]:]
        aS = solve_reduced(s_p, r_p)
        aS = np.roll(aS, aS.shape[1]//2, axis=1)

        gramm = wedge_dense_d(aS, aS, self.d)
        gramm_l = np.tril(gramm)

        i_stab = self.i_stabilizers()
        aS = np.mod(aS + gramm_l.T @ i_stab.todense(), self.d)

        return np.array(aS, dtype=np.uint8)
