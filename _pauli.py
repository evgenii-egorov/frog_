import numpy as np
import scipy.sparse as sp


def pauli_to_bs(pauli_string):
    """
    XII -> (100|000) -- (X_on_qubit1,X_on_qubit2,X_onqubit3 | Z_on_qubit1,Z_on_qubit2,Z_onqubit3)
    YII -> (100|100)
    XZZ -> (100|011)
    """
    xs = np.zeros(len(pauli_string), dtype=np.uint8) 
    zs = np.zeros(len(pauli_string), dtype=np.uint8)
    for i, p in enumerate(pauli_string):
        x = (p == 'X') or (p == 'Y')
        z = (p == 'Z') or (p == 'Y')
        xs[i], zs[i] = x, z
    return xs, zs # xs, zs are binary bits (100|000) recording whether or not (X|Z) operate on each qubit


def v_pauli_to_bs(pauli):
    if isinstance(pauli, str):
        a_px, a_pz = pauli_to_bs(pauli)
    else: # pauli can be a series of strings [XIY, XXZ, YI]
        m_len = 0
        for p in pauli:
            m_len = max(m_len, len(p))
        a_px = np.zeros((len(pauli), m_len), dtype=np.uint8) # 
        a_pz = np.zeros((len(pauli), m_len), dtype=np.uint8)
        for i, p in enumerate(pauli):
            xs, zs = pauli_to_bs(p)
            a_px[i, :len(xs)] = xs
            a_pz[i, :len(zs)] = zs
    return np.hstack((a_px, a_pz))


def bs_to_pauli(bs):
    """
    (101|001) -> XIY
    """
    trans_dict = str.maketrans('0123', 'IXZY')
    xs, zs = np.hsplit(bs, 2)
    xs_s = xs + 2 * zs
    b_str = ''.join(xs_s.astype(str))
    return b_str.translate(trans_dict)


def v_bs_to_pauli(b):
    return bs_to_pauli(b) if isinstance(b, str) else [bs_to_pauli(bs) for bs in b]


def wedge(ab, AB): 
    """
    (a|b)wedge(A|B) = <(b|a), (A|B)>
    """
    N = ab.shape[1] // 2
    a, b = ab[:, :N], ab[:, N:]
    ba = sp.hstack((b, a), format="csr", dtype=np.uint8)
    result = ba @ AB.T
    result.data = np.mod(result.data, 2)
    result.eliminate_zeros()
    return result


def wedge_dense(ab, AB):
    """
    (a|b)wedge(A|B) = <(b|a), (A|B)>
    """
    return np.mod(np.roll(ab, ab.shape[1] // 2, axis=1) @ AB.T, 2)

def wedge_d(ab, AB, d): # only works when ab and AB are 2d sparse arrays
    """
    (a|b)wedge(A|B) = <(b|a), (A|B)>
    """
    inv = np.mod(-1,d)
    N = ab.shape[1] // 2
    a, b, A, B = ab[:, :N], ab[:, N:], AB[:, :N], AB[:, N:]
    a, b, A, B = sp.csr_matrix(a), sp.csr_matrix(b), sp.csr_matrix(A), sp.csr_matrix(B) 
    result = a @ B.T + b @ A.T.multiply(inv)
    result.data = np.mod(result.data, d)
    result.eliminate_zeros()
    return result 


def wedge_dense_d(ab, AB, d):
    """
    (a|b)wedge(A|B) = <(b|a), (A|B)>
    """
    inv = np.mod(-1,d)
    N = ab.shape[1] // 2
    a, b = ab[:, :N], ab[:, N:]
    A, B = AB[:, :N], AB[:, N:]
    return np.mod(a @ B.T + b @ A.T*inv, d)


def J_wedge(ab, AB):
    raise NotImplementedError  # from scipy import sparse, kron?


def bs(ab, AB):
    if isinstance(ab, np.ndarray):
        r = np.mod(ab + AB, 2)
    else:
        r = ab + AB
        r.data = np.mod(r.data, 2)
        r.eliminate_zeros()
    return r


