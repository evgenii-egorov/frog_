import abc
import torch
import torch.nn as nn


class LearnableDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, L, g, pooling, func):
        super().__init__()
        self.L = L
        self.g = g
        self.lib = self.g.lib
        self.pooling = pooling
        self.pooling.__init__(L, g)
        self.basic_function = func

    def adjust_L(self, L):
        self.L = L
        self.g.adjust_L(L)
        self.pooling.__init__(L, self.g)
        for m in self.basic_function.modules():
            if hasattr(m, 'L'):
                m.__init__(L)


    @abc.abstractmethod
    def decode2all(self, syndrome):
        pass

    def forward(self, syndrome):
        out = self.decode2all(syndrome)
        out = self.pooling.forward(out, syndrome)
        return out

    def decode2label(self, syndrome):
        logits = self.forward(syndrome)

        b_size = logits.shape[0]
        logits = logits.reshape(b_size, 16)
        _, prediction = torch.max(logits, 1)
        return prediction


class SpatialAverage(metaclass=abc.ABCMeta):
    def __init__(self, L, g):
        self.L = L
        self.g = g
        self.lib = self.g.lib

    @abc.abstractmethod
    def forward(self, x, syndrome):
        pass


class BasicInftyDecoder(LearnableDecoder):
    def decode2all(self, syndrome):
        b_size = syndrome.shape[0]
        out = self.lib.zeros((b_size, self.L, self.L, 4, 2, 2))
        for i in range(self.L):
            for j in range(self.L):
                x = self.g.syndrome_action(syndrome, direction='x', power=i)
                x = self.g.syndrome_action(x, direction='y', power=j)
                out[:, j, i,:,:,:] = self.basic_function(x)
        return out


class Basic1Decoder(LearnableDecoder):
    def decode2all(self, syndrome):
        out = self.basic_function(syndrome)
        return out


class L2Projection(SpatialAverage):
    def logic_action_average(self, x, syndrome, direction):
        flag_p, flag_d = self.g.condition_logic_action_composite(syndrome, direction=direction)

        b_size = x.shape[0]
        x = x.reshape(b_size * self.L, self.L, 4, 2, 2)
        for i in range(self.L):
            flag_p_r = self.lib.tile(flag_p[:, i][:, None], self.L).reshape(b_size * self.L)
            flag_d_r = self.lib.tile(flag_d[:, i][:, None], self.L).reshape(b_size * self.L)

            x[flag_p_r,i,:,:,:] = self.g.primal_non_id_logic_action(x[flag_p_r,i,:,:,:], direction=direction)
            x[flag_d_r,i,:,:,:] = self.g.dual_non_id_logic_action(x[flag_d_r,i,:,:,:], direction=direction)

        x = x.reshape(b_size, self.L, self.L, 4, 2, 2)
        return x

    def forward(self, x, syndrome):
        x = self.logic_action_average(x, syndrome, 'x')

        x = self.lib.swapaxes(x, 1, 2)
        x = self.logic_action_average(x, syndrome, 'y')
        x = self.lib.swapaxes(x, 1, 2)

        x = self.lib.mean(x, axis=(1, 2))
        return x


class AveragePooling(SpatialAverage):
    def forward(self, x, syndrome):
        x = self.lib.mean(x, axis=(1,2))
        return x


class NoPooling(SpatialAverage):
    def forward(self, x, syndrome):
        return x
