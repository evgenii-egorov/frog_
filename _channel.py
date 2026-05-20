import abc
import numpy as np
from tqdm import tqdm


class Channel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, b_size):
        return 
        
    @abc.abstractmethod
    def log_prob(self, input):
        return
    

class Depolarising(Channel):
    def __init__(self, n, xzy_p, rng=None):
        self.n = n
        self.rng = np.random.default_rng() if rng is None else rng

        self.xzy  = xzy_p if type(xzy_p) is tuple else (xzy_p/3., xzy_p/3., xzy_p/3.)
        p_e = self.xzy[0] + self.xzy[1] + self.xzy[2]
        if p_e > 1:
            raise NotImplementedError
        else:
            self.i =  1 - p_e

    def sample(self, b_size):
        n = self.n
        x, z, y = self.xzy
        i =  self.i

        noise = self.rng.choice(['I', 'X', 'Z', 'Y'], size=n * b_size, p=[i, x, z, y]).reshape(b_size, n)

        xzy = np.zeros((2, b_size, n),  dtype=np.uint8)
        xzy[0][np.where((noise == 'X') | (noise == 'Y'))] = 1
        xzy[1][np.where((noise == 'Z') | (noise == 'Y'))] = 1
        out = np.hstack([xzy[0], xzy[1]])
        return out 
    
    def log_prob(self, input, common_factor=True):
        batch = input.astype(float)

        n = self.n
        x, z, y = self.xzy
        log_i =  np.log(self.i)

        log_p = [np.log(p) -  log_i if p > 0 else 0 for p in [x, z, y]]
        log_p_x, log_p_z, log_p_y = log_p[0], log_p[1], log_p[2]

        y_count = np.sum(batch[:, :n] * batch[:, n:], axis=1).astype(float)
        x_count = np.sum(batch[:, :n], axis=1).astype(float) - y_count
        z_count = np.sum(batch[:, n:], axis=1).astype(float) - y_count

        log_prob = x_count * log_p_x + z_count * log_p_z + y_count * log_p_y

        mask_x = (self.xzy[0] == 0) and (self.xzy[2] == 0) and (x_count + y_count > 0)
        mask_z = (self.xzy[1] == 0) and (self.xzy[2] == 0) and (z_count + y_count > 0)
        mask = mask_x | mask_z
        log_prob[mask] = - np.infty

        if common_factor:
            log_prob += self.n * log_i
            
        return log_prob


class Independent(Channel):       
    def __init__(self, n, d, xj_p, zj_p, rng=None): 
        '''
        xj_p is probability of X^j error for 1 < j <= d-1, x_p is the total prob of an x error happening
        '''
        self.n = n
        self.d = d
        self.rng = np.random.default_rng() if rng is None else rng
        self.xj  = xj_p if type(xj_p) is tuple else tuple([xj_p/(d-1)]*(d-1))
        x_p = sum(self.xj) 
        if x_p >= 1:
            raise NotImplementedError
        else:
            self.xi =  1 - x_p
        self.zj  = zj_p if type(zj_p) is tuple else tuple([zj_p/(d-1)]*(d-1))
        z_p = sum(self.zj) 
        if z_p >= 1:
            raise NotImplementedError
        else:
            self.zi =  1 - z_p
    
    def sample(self, b_size):
        n = self.n
        d = self.d
        self.b = b_size
        xi,zi =  self.xi, self.zi
        self.x_prob = [xi]+list(self.xj)
        self.z_prob = [zi]+list(self.zj)
        self.noise_list = list(range(d))
        x_noise = self.rng.choice(self.noise_list, size=n * b_size, p=self.x_prob).reshape(b_size, n) 
        z_noise = self.rng.choice(self.noise_list, size=n * b_size, p=self.z_prob).reshape(b_size, n) 
        return np.hstack([x_noise,z_noise])
    
    def log_prob(self, sample, debug=False, counter=False):
        batch = sample.astype(float)
        log_xp = [np.log(p) for p in self.x_prob]
        log_zp = [np.log(p) for p in self.z_prob]

        x_error_count = np.zeros((self.b,self.d))
        for m in tqdm(range(self.b)):
            for n in range(self.d):
                x_error_count[m][n] = batch[m,:self.n].tolist().count(n)
        z_error_count = np.zeros((self.b,self.d))
        for m in tqdm(range(self.b)):
            for n in range(self.d):
                z_error_count[m][n] = batch[m,self.n:].tolist().count(n)

        log_xp_tot = np.zeros(self.b)
        log_zp_tot = np.zeros(self.b)
        for m in tqdm(range(self.b)):
            log_xp_tot[m] = sum(x_error_count[m]*log_xp) 
            log_zp_tot[m] = sum(z_error_count[m]*log_zp)

        if not debug:
            return log_xp_tot + log_zp_tot
        if counter == True:
            return x_error_count, z_error_count
        return np.hstack([log_xp_tot[np.newaxis].T,log_zp_tot[np.newaxis].T])


def test_sample_large_batch():
    noise = Independent(3,5,0.4,0.4)
    b_size = 1000000
    large_batch = noise.sample(b_size)
    x_counter, z_counter= noise.log_prob(large_batch,debug=True,counter=True)
    x_count = x_counter.sum(axis=0)
    z_count = z_counter.sum(axis=0)
    for i in range(noise.d):
        freq_minus_prob = (x_count[i]/b_size - noise.x_prob[i]) /b_size
        assert -0.001 <= freq_minus_prob <= 0.001, f"less than 0.001 difference in x_prob and freq expected, got: {freq_minus_prob}"
    for i in range(noise.d):
        freq_minus_prob = (z_count[i]/b_size - noise.z_prob[i]) /b_size
        assert -0.001 <= freq_minus_prob <= 0.001, f"less than 0.001 difference in z_prob and freq expected, got: {freq_minus_prob}"


def test_sample_zero():
    noise = Independent(3,5,0,0)
    sample = noise.sample(1000000)
    shape = sample.shape
    for i in tqdm(range(shape[0])):
        for j in range(shape[1]):
            assert sample[i][j] == 0 

def test_log_prob_all_equal():
    noise = Independent(3,5,0.8,0.8)
    sample = noise.sample(10000)
    prob = noise.log_prob(sample)
    for i in tqdm(prob):
        assert abs(i - np.log(0.2**6)) < 0.0000001
