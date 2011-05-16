#!/usr/bin/python2
import random
from operator import lt
from functools import partial
import numpy as np
import matplotlib.pylab as plot


def main():
    """
    Lotka-Volterra system:
        dx/dt = ax - xy
        dy/dt = bxy - cy

        where
            a(t) = a_0 * (1 + sin(omega * t))
            0 < b < 1
            0 < c

    Migrations:
        dx_i/dt = nu_x * Sum(x_j - x_i)
        dy_i/dt = nu_y * Sum(y_j - y_i)

    """
    b = 0.5
    c = 1
    a_0 = 1
    omega = 1
    nu_x = 1
    nu_y = 1
    geom_w = 3
    geom_h = 3
    dt = 0.001

    choices = range(5, 100, 5)
    getr = lambda w: [random.choice(choices) for i in xrange(w)]
    getm = lambda w, h: [getr(w) for i in xrange(h)]
    x_0 = np.array(getm(geom_w, geom_h), dtype=np.int32)
    y_0 = np.array(getm(geom_w, geom_h), dtype=np.int32)

    t, X, Y = solve(b, c, a_0, omega, nu_x, nu_y, geom_w, geom_h, dt, x_0, y_0)

    plot.grid(True)
    plot.xlabel('Time')
    plot.ylabel('Amount')
    p1, p2 = plot.plot(t, x, 'g-', t, y, 'r-')
    plot.legend([p1, p2], ['prey', 'predators'])
    plot.show()


def solve(b, c, a_0, omega, nu_x, nu_y, w, h, dt, x_0, y_0):
    T = [0]
    X = [x_0]
    Y = [y_0]
    for i in xrange(3000):
        t = dt * (i + 1)
        a = a_0 * (1 + np.sin(omega * t))
        x, y = solve_step(a, b, c, nu_x, nu_y, w, h, dt, X[i], Y[i])
        map(lambda (A, a): A.append(a), [(T, t), (X, x), (Y, y)])
    return T, map(lambda xs: xs.sum(), X), map(lambda ys: ys.sum(), Y)


odd = lambda n: n % 2 == 0
so = lambda n, w: [n-w-w, n-w-1, n-w, n+w-1, n+w, n+w+w]
se = lambda n, w: [n-w-w, n-w, n-w+1, n+w, n+w+1, n+w+w]
siblings = lambda n, w: filter(partial(lt, 0), so(n, w) if odd(n) else se(n, w))


def solve_step(a, b, c, nu_x, nu_y, w, h, dt, x, y):
    n_siblings = lambda n: len(siblings(n, w))
    n_to_rc = lambda n: divmod(n, w)
    dt_nu_x = dt * nu_x
    dt_nu_y = dt * nu_y
    k1_x = lambda n: dt * (a - y[n_to_rc(n)]) - 1 - n_siblings(n) * dt_nu_x
    k2_x = lambda n: dt_nu_x
    k1_y = lambda n: dt * (b * x[n_to_rc(n)] - c) - 1 - n_siblings(n) * dt_nu_y
    k2_y = lambda n: dt_nu_y
    x = solve_one(k1_x, k2_x, w, h, x)
    y = solve_one(k1_y, k2_y, w, h, y)
    return x, y


def solve_one(k1, k2, w, h, v):
    rc_to_n = lambda r, c: r * w + c
    def geta(i, j, sibs):
        n = rc_to_n(i, j)
        return k1(n) if i == j else (k2(n) if n in sibs else 0)
    rank = xrange(w * h)
    getr = lambda i: [geta(i, j, siblings(rc_to_n(i, i), w)) for j in rank]
    getm = lambda: [getr(i) for i in rank]
    A = np.array(getm(), np.float)
    B = v
    return np.linalg.solve(A, B)


if __name__ == '__main__':
    main()
