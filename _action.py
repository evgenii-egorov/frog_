import abc
import numpy as np
import _numpy2torch as np2torch


class SymmetryAction(metaclass=abc.ABCMeta):
    def __init__(self, L, mode_numpy=True):
        self.L = L
        self.lib = np if mode_numpy else np2torch

    def adjust_L(self, L):
        self.L = L

    @abc.abstractmethod
    def simple_chain_action(self, s_chain, **kwargs):
        pass

    @abc.abstractmethod
    def simple_syndrome_action(self, s_syndrome, **kwargs):
        pass

    @abc.abstractmethod
    def primal_non_id_logic_action(self, probs, **kwargs):
        pass

    @abc.abstractmethod
    def condition_logic_action(self, s_syndrome, **kwargs):
        pass

    @abc.abstractmethod
    def condition_logic_action_all_degree(self, s_syndrome, **kwargs):
        pass

    def composite_object_action(self, c_object, action):
        p1, p2 = self.lib.hsplit(c_object, 2)
        g_p1, g_p2 = [action(o) for o in [p1, p2]]
        return self.lib.hstack((g_p1, g_p2))

    def syndrome_action(self, syndrome, **kwargs):
        action = lambda s: self.simple_syndrome_action(s_syndrome=s, **kwargs)
        g_syndrome = self.composite_object_action(syndrome, action)
        return g_syndrome

    def chain_action(self, chain, **kwargs):
        action = lambda c: self.simple_chain_action(s_chain=c, **kwargs)
        g_chain = self.composite_object_action(chain, action)
        return g_chain

    def swap_primal_dual_logic(self, probs):
        b_size = probs.shape[0]
        out = self.lib.zeros_like(probs)
        for i in range(4):
            m, n = i // 2, i % 2
            out[:, i, :, :] = probs[:, :, n, m].reshape(b_size, 2, 2)

        out = out[:, [0, 2, 1, 3], :, :]
        return out

    def dual_non_id_logic_action(self, probs, **kwargs):
        out = self.lib.copy(probs)
        kwargs['direction'] = 'y' if kwargs['direction'] == 'x' else 'x'

        out = self.swap_primal_dual_logic(out)
        out = self.primal_non_id_logic_action(out, **kwargs)
        out = self.swap_primal_dual_logic(out)

        kwargs['direction'] = 'y' if kwargs['direction'] == 'x' else 'x'
        return out

    def logic_action(self, probs, c_syndrome, **kwargs):
        out = self.lib.copy(probs)
        p_s, d_s = self.lib.hsplit(c_syndrome, 2)

        kwargs['primal'] = True
        mask = self.condition_logic_action(p_s, **kwargs)
        out[mask] = self.primal_non_id_logic_action(out, **kwargs)[mask]

        kwargs['primal'] = False
        mask = self.condition_logic_action(d_s, **kwargs)
        out[mask] = self.dual_non_id_logic_action(out, **kwargs)[mask]

        return out


class Translation(SymmetryAction):
    @staticmethod
    def ax4action_from_direction(direction):
        if direction == 'x':
            ax = 1 + 1
        elif direction == 'y':
            ax = 0 + 1
        else:
            raise NotImplementedError
        return ax

    def make_square(self, o):
        b_size = o.shape[0]
        return o.reshape(b_size, self.L, self.L)

    def translate(self, o, direction, power, keepdims=True):
        b_size = o.shape[0]
        axis = self.ax4action_from_direction(direction)
        out = self.lib.roll(self.make_square(o), shift=power, axis=axis)
        out = out.reshape(b_size, -1) if keepdims else out
        return out

    def simple_chain_action(self, s_chain, **kwargs):
        h, v = self.lib.hsplit(s_chain, 2)
        gh, gv = [self.translate(s, **kwargs) for s in [h, v]]
        return self.lib.hstack((gh, gv))

    def simple_syndrome_action(self, s_syndrome, **kwargs):
        g_s = self.translate(s_syndrome, **kwargs)
        return g_s

    def primal_non_id_logic_action(self, probs, **kwargs):
        out = self.lib.copy(probs)
        if kwargs['direction'] == 'y':
            out = self.lib.roll(probs, shift=1, axis=-2)
        elif kwargs['direction'] == 'x':
            out = self.lib.roll(probs, shift=1, axis=-1)
        else:
            raise NotImplementedError
        return out

    def condition_logic_action(self, s_syndrome, **kwargs):
        condition_array = self.condition_logic_action_all_degree(s_syndrome, **kwargs)
        power = kwargs['power'] % self.L
        condition = condition_array[:, power]
        return condition

    def condition_logic_action_all_degree(self, s_syndrome, **kwargs):
        ref = -1 if kwargs['primal'] else 0
        ax4sum = 0 + 1 if kwargs['direction'] == 'x' else 1 + 1
        r_syndrome = self.translate(s_syndrome, kwargs['direction'], power=-ref, keepdims=False)
        r_syndrome = self.lib.sum(r_syndrome, axis=ax4sum)

        r_syndrome = self.lib.roll(self.lib.flip(r_syndrome, axis=-1), shift=1, axis=-1)
        condition_array = self.lib.cumsum(r_syndrome, axis=-1)
        condition_array = self.lib.roll(condition_array, shift=1, axis=-1)
        condition_array = self.lib.mod(condition_array, 2) > 0

        return condition_array

    def condition_logic_action_composite(self, c_syndrome, **kwargs):
        p_s, d_s = self.lib.hsplit(c_syndrome, 2)
        kwargs['primal'] = True
        flags_primal = self.condition_logic_action_all_degree(p_s, **kwargs)
        kwargs['primal'] = False
        flags_dual = self.condition_logic_action_all_degree(d_s, **kwargs)
        return flags_primal, flags_dual






