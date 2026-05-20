import drawSvg as draw
import numpy as np


class Vertex:
    def __init__(self, primal, a, b):
        self.shift = 0.0 if primal else 0.5
        self.primal = primal
        self.x = a + self.shift
        self.y = - b - self.shift
        self.color = ['#0d976f', '#393939']
        self.active_color = ['#ff6da3', '#048ce9']
        self.id = str(a) + str(b) + str(primal * 1)

    def draw_v(self, excited=False):
        circ = draw.Circle(self.x, self.y, r='0.1', fill=self.color[self.primal],
                           stroke_width='0.01', id=self.id)
        text = draw.Text(text=self.id, x=self.x - 0.075, y=self.y,
                         fill='white', fontSize=0.08)
        out = [circ, text]

        if excited:
            circ_excited = draw.Circle(self.x, self.y, r='0.5',
                                       fill=self.active_color[not self.primal],
                                       stroke_width='0.01', opacity='0.25')
            out += [circ_excited]
        return out

    def draw_edge(self, h, v, active):
        sx = self.x
        sy = self.y
        primal = self.primal

        if h:
            sx = sx + (0.1 if primal else - 0.1)
            ey = sy
            ex = sx + (0.8 if primal else - 0.8)
        elif v:
            sy = sy - (0.1 if primal else - 0.1)
            ex = sx
            ey = sy - (0.8 if primal else - 0.8)
        else:
            raise ValueError('Edge should h or v')

        if active:
            color = self.active_color[self.primal]
            line = draw.Line(sx=sx, sy=sy, ex=ex, ey=ey,
                             stroke=color, stroke_width='0.05', opacity='0.6')
        else:
            color = self.color[self.primal]
            line = draw.Line(sx=sx, sy=sy, ex=ex,
                             ey=ey, stroke_width='0.01', stroke=color)

        return [line]


def draw_simple(simple_string, L, primal):
    if primal:
        h_q, v_q = np.hsplit(simple_string, 2)
    else:
        h_q, v_q = np.hsplit(simple_string, 2)
        v_q, h_q = h_q, v_q

    out_ls = []
    grid = [(a, b) for a in range(L) for b in range(L)]
    for xy in grid:
        v = Vertex(primal, *xy)
        x, y = xy

        c = y * L + x
        h_edge_status = True if h_q[c] > 0 else False
        v_edge_status = True if v_q[c] > 0 else False

        out_ls += v.draw_edge(h=True, v=False, active=False)
        out_ls += v.draw_edge(h=False, v=True, active=False)

        out_ls += v.draw_edge(h=True, v=False, active=h_edge_status)
        out_ls += v.draw_edge(h=False, v=True, active=v_edge_status)
    return out_ls


def draw_syndrome(syndrome, L):
    out_ls = []
    excited_primal_f, excited_dual_f = np.hsplit(syndrome, 2)
    grid = [(a, b) for a in range(L) for b in range(L)]
    for xy in grid:
        x, y = xy
        c = y * L + x

        v_status = True if excited_dual_f[c] > 0 else False
        v_dual_status = True if excited_primal_f[c] > 0 else False

        v = Vertex(True, *xy)
        v_dual = Vertex(False, *xy)

        out_ls += v.draw_v(excited=v_status)
        out_ls += v_dual.draw_v(excited=v_dual_status)
    return out_ls


def show_plot(object_list, L, scale=128):
    out = draw.Drawing(1 + L, 1 + L, origin=(-0.5, -L - 0.5))
    out_ls = []

    grid = [(a, b) for a in range(L) for b in range(L)]
    for xy in grid:
        v, v_ = Vertex(True, *xy), Vertex(False, *xy)
        out_ls += v.draw_v()
        out_ls += v_.draw_v()

    out_ls = out_ls + object_list
    for i in out_ls: out.append(i)
    out.setPixelScale(scale)
    out.rasterize()
    return out


def plot_pauli(xz_string, L):
    x, z = np.hsplit(xz_string, 2)
    primal = draw_simple(z, L, True)
    dual = draw_simple(x, L, False)
    return primal + dual


def plot_primal_dual(xz_string, L):
    x, z = np.hsplit(xz_string, 2)
    primal = draw_simple(z, L, True)
    dual = draw_simple(x, L, False)

    primal_star = draw_simple(x, L, True)
    dual_star = draw_simple(z, L, False)
    return primal + dual + primal_star + dual_star


def plot_subspace(subspace, L):
    sub_ls = sum([plot_pauli(i, L) for i in subspace], [])
    return show_plot(sub_ls, L)


import cairosvg
from PIL import Image
from io import BytesIO


def svg_object2matplotlib(obj_svg):
    svg2png = cairosvg.svg2png(obj_svg.asSvg())
    img = Image.open(BytesIO(svg2png))
    return img


class CosetProbsPrinting:
    @staticmethod
    def coset_prob_4x4(probs):
        m = np.block([
            [probs[0], probs[1]],
            [probs[2], probs[3]]])
        return m

    @staticmethod
    def draw_g_probs(original_probs, g_syndrome_probs, g_original_probs, ax):
        title = ['Pr(Syndrome)', 'Pr(G Syndrome)', 'G Pr(Syndrome)']
        m = [CosetProbsPrinting.coset_prob_4x4(i) for i in [original_probs, g_syndrome_probs, g_original_probs]]

        for i, a in enumerate(ax.flatten()):
            a.matshow(m[i], cmap='tab10')
            a.set_title(title[i])
            a.axis('off')

            for j in [0, 1, 2, 3]:
                wd = [4, 1][j % 2]
                a.hlines(y=j-0.5, xmin=-0.5, xmax=4-0.4, color='white', linewidth=wd)
                a.vlines(x=j-0.5, ymin=-0.5, ymax=4-0.4, color='white', linewidth=wd)

            for k in range(4):
                for l in range(4):
                    c = np.round(m[i][l, k], 3)
                    a.text(k, l, str(c), va='center', ha='center',
                           color='black', fontweight='bold')

    @staticmethod
    def draw_probs(probs, ax, title=None):
        m = CosetProbsPrinting.coset_prob_4x4(probs)
        ax.matshow(m.astype(float), cmap='tab10')
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
        
        for j in [0, 1, 2, 3]:
                wd = [4, 1][j % 2]
                ax.hlines(y=j-0.5, xmin=-0.5, xmax=4-0.4, color='white', linewidth=wd)
                ax.vlines(x=j-0.5, ymin=-0.5, ymax=4-0.4, color='white', linewidth=wd)
        
        for k in range(4):
            for l in range(4):
                c = m[l, k] if type(m[l,k]) is not float else np.round(m[l, k], 5)
                ax.text(k, l, str(c), va='center', ha='center',
                        color='black', fontweight='bold')

