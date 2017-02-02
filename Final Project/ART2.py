import numpy as np
import math

__author__ = "Michael Suggs // mjs3607@uncw.edu"


class ART2:
    """ 1-line summary.

    Long summary here.

    Attributes:
        a.
        b.
        c.
    """

    def __init__(self, inputs, num_cat, nsteps, rho=0.9, a=5.0, b=5.0, c=0.225,
                 d=0.8, theta=0.3, ac_dif=0.001, z_dif=0.01):

        self.inputs = [np.asarray(i) for i in inputs]
        self.num_in = self.inputs[0].size
        self.num_pat = len(self.inputs)

        self.params = {'rho' : rho, 'a' : a, 'b' : b, 'c' : c, 'd' : d,
                       'theta' : theta, 'ac_dif' : ac_dif, 'z_dif' : z_dif,
                       'nsteps' : nsteps}

        self.x = np.zeros(self.num_in)
        self.w = np.zeros(self.num_in)
        self.u = np.zeros(self.num_in)
        self.p = np.zeros(self.num_in)
        self.v = np.zeros(self.num_in)
        self.q = np.zeros(self.num_in)

        self.xp = np.zeros(self.num_in)
        self.wp = np.zeros(self.num_in)
        self.up = np.zeros(self.num_in)
        self.pp = np.zeros(self.num_in)
        self.vp = np.zeros(self.num_in)
        self.qp = np.zeros(self.num_in)

        self.r = np.zeros(self.num_in)
        self.y = np.zeros(self.num_in)
        self.mismatch = np.zeros(self.num_in)
        self.num_mismatched = 0

        self.bottom_up = np.zeros((2, self.num_in))
        self.top_down = np.zeros((2, self.num_in))

        self.encoded_pat = np.zeros(num_cat)
        self.active_F2 = -1
        self.learning = 1
        self.resonating = False

        # get_inputs()
        # ? alloc_pops()
        # bottomup_reset()
        # topdown_reset()

    def signal_fn(self, x):
        if 0 <= x < self.params['theta']:
            return 0
        else:
            return x

    def rk4(self, step_size, x, z_eq1, z_eq2):
        """ A method of numerically integrating ordinary differential equations
          by using a trial step at the midpoint of an interval to cancel out
          lower-order error terms.

        :param step_size:
        :param x:
        :param z_eq1:
        :param z_eq2:
        :return:
        """

        def fn(z):
            return z_eq1 * (z_eq2 - z)

        k1 = step_size * fn(x)
        k2 = step_size * fn(x + (k1 / 2))
        k3 = step_size * fn(x + (k2 / 2))
        k4 = step_size * fn(x + k3)
        y = (x + k1 + (2 * k2) + (2 * k3) + k4) / 6

        return y

    def bottom_up_reset(self):
        for i in range(self.num_in):
            for j in range(self.encoded_pat.size):
                self.bottom_up[i][j] = 1 / ((1 - self.params['d']
                                             * math.sqrt(self.num_in)))

    def top_down_reset(self):
        for j in range(self.encoded_pat.size):
            for i in range(self.num_in):
                self.top_down[j][i] = 0

    def update_z(self):
        bu_prev = np.zeros(self.num_in)
        delta_bu = 0
        td_prev = np.zeros(self.num_in)
        delta_td = 0
        step_size = 0.1

        for i in range(self.num_in):
            bu_prev[i] = self.bottom_up[i][self.active_F2]
            td_prev[i] = self.top_down[self.active_F2][i]

            z_eq1 = self.params['d'] * (1 - self.params['d'])
            z_eq2 = self.u[i] / (1 - self.params['d'])

            self.bottom_up[i][self.active_F2] = self.rk4(
                step_size, self.bottom_up[i][self.active_F2], z_eq1, z_eq2)
            self.top_down[self.active_F2][i] = self.rk4(
                step_size, self.top_down[self.active_F2][i], z_eq1, z_eq2)

            delta_bu += abs(bu_prev[i] - self.bottom_up[i][self.active_F2])
        delta_td = np.subtract(self.top_down[self.active_F2], td_prev)

        return delta_bu + delta_td


    def art_cycle(self, training=True):
        self.learning = 1 if training is True else 0

        for i in range(self.mismatch):
            self.mismatch[i] = 0
        self.num_mismatched = 0

        for i in range(self.num_pat):
            for j in range(self.num_in):
                # reset F0, F1 nodes for new pattern
                self.active_F2 = -1

                self.update_F0(i)
                self.update_F1(i)

    def update_F0(self, p_num):
        count = 0
        wp_prev = np.zeros(self.num_in)
        delta_v = 1.0

        while (delta_v > self.params['ac_dif'] and
                       count < self.params['nsteps']) or count < 2:

            for i in range(self.num_in):
                wp_prev[i] = self.wp[i]
                self.wp[i] = self.inputs[p_num][i]
                self.pp[i] = self.up[i]

            w_norm = np.linalg.norm(self.wp)
            w_norm = w_norm if w_norm > 0.01 else 1.0
            p_norm = np.linalg.norm(self.pp)
            p_norm = p_norm if p_norm > 0.01 else 1.0

            for i in range(self.num_in):
                self.qp[i] = self.pp[i] / p_norm
                self.xp[i] = self.wp[i] / w_norm
                self.vp[i] = (self.signal_fn(self.xp[i]) + self.params['b']
                              * self.signal_fn(self.qp[i]))

            v_norm = np.linalg.norm(self.vp)
            v_norm = v_norm if v_norm > 0.01 else 1.0

            for i in range(self.num_in):
                self.up[i] = self.vp[i] / v_norm

            delta_v = np.subtract(wp_prev, self.wp)
            count += 1

    def update_F1(self, p_num):
        count = 0
        w_prev = np.zeros(self.num_in)
        p_prev = np.zeros(self.num_in)
        delta_v = 1.0
        delta_z = 0.0

        while ((delta_v > self.params['ac_dif'] and
                        count < self.params['nsteps']) or count < 2 or
                       delta_z > self.params['z_dif']):
            for i in range(self.num_in):
                w_prev = self.w[i]
                p_prev = self.p[i]
                self.w[i] = self.qp[i] + self.params['a'] * self.u[i]
                sum_gyz = (0 if self.active_F2 < 0 else self.params['d'] *
                                            self.top_down[self.active_F2][i])
                self.p[i] = self.u[i] + sum_gyz

            w_norm = np.linalg.norm(self.w)
            w_norm = w_norm if w_norm > 0.01 else 1.0
            p_norm = np.linalg.norm(self.p)
            p_norm = p_norm if p_norm > 0.01 else 1.0

            for i in range(self.num_in):
                self.q[i] = self.p[i] / p_norm
                self.x[i] = self.w[i] / w_norm
                self.v[i] = (self.signal_fn(self.x[i]) + self.params['b']
                             * self.signal_fn(self.q[i]))

            v_norm = np.linalg.norm(self.v)
            v_norm = v_norm if v_norm > 0.01 else 1.0

            for i in range(self.num_in):
                self.u[i] = self.v[i] / v_norm

            delta_v = np.subtract(w_prev, self.w) + np.subtract(p_prev, self.p)
            delta_z = 0.0

            # Resonates when winner is found - encode the resonating pattern
            if not self.resonating:
                self.update_F2()
            else:
                self.encoded_pat[p_num] = self.active_F2

            # Reset F2 if mismatch occurs (R exceeds the vigilance)
            if self.update_r() < self.params['rho']:
                self.reset_F2(p_num)
                count = 0
                self.resonating = False
            else:
                delta_z = self.update_z()
            count += 1
        self.resonating = False

    def update_F2(self):
        max_node = 0

        for i in range(self.encoded_pat.size):
            self.y[i] = 0
            if self.mismatch[i] > -1:
                for j in range(self.num_in):
                    self.y[i] += self.bottom_up[j][i] * self.p[j]
            max_node = self.y[i] if max_node < self.y[i] else max_node

        if max_node is 0:
            self.active_F2 = -1
            return

        for i in range(self.encoded_pat.size):
            if self.y[i] >= max_node:
                self.active_F2 = i
                self.resonating = True
                break

    def reset_F2(self, p_num):
        print("Pattern no. {} caused an F2 reset in node {}.".format(
            p_num, self.active_F2))

        self.y[self.active_F2] = 0
        self.mismatch[self.active_F2] = -1
        self.num_mismatched += 1
        self.resonating = False
        self.active_F2 = -1
        self.encoded_pat[p_num] = -1

        for i in range(self.num_in):
            self.w[i] = 0
            self.p[i] = 0
            self.x[i] = 0
            self.q[i] = 0
            self.v[i] = 0
            self.u[i] = 0

    def update_r(self):
        p_norm = np.linalg.norm(self.p)
        qp_norm = np.linalg.norm(self.qp)
        u_norm = np.linalg.norm(self.u)

        for i in range(self.num_in):
            self.r[i] = ((self.qp[i] + self.params['c'] * self.p[i]) /
                         (self.params['c'] * p_norm + qp_norm))

        return np.linalg.norm(self.r)
