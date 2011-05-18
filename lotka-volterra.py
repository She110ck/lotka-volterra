#!/usr/bin/python2
import random
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
    getm = lambda w, h: [random.choice(choices) for i in range(w * h)]
    x_0 = np.array(getm(geom_w, geom_h), dtype=np.int32)
    y_0 = np.array(getm(geom_w, geom_h), dtype=np.int32)

    t, x, y = solve(b, c, a_0, omega, nu_x, nu_y, geom_w, geom_h, dt, x_0, y_0)

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


l = lambda n, w: (n / w) % 2 == 0
edge = lambda n, w: n % w == 0 and l(n, w) or n % w == w - 1 and not l(n, w)
all_siblings = lambda n, w: ([n-w-w, n-w, n+w, n+w+w] if edge(n, w)
                        else [n-w-w, n-w-1, n-w, n+w-1, n+w, n+w+w] if l(n, w)
                        else [n-w-w, n-w, n-w+1, n+w, n+w+1, n+w+w])
siblings = lambda n, w, h: filter(lambda s: 0 <= s < w * h, all_siblings(n, w))


def solve_step(a, b, c, nu_x, nu_y, w, h, dt, x, y):
    n_siblings = lambda n: len(siblings(n, w, h))
    dt_nu_x = dt * nu_x
    dt_nu_y = dt * nu_y
    k1_x = lambda n: dt * (a - y[n]) - 1 - n_siblings(n) * dt_nu_x
    k2_x = lambda n: dt_nu_x
    k1_y = lambda n: dt * (b * x[n] - c) - 1 - n_siblings(n) * dt_nu_y
    k2_y = lambda n: dt_nu_y
    x = solve_one(k1_x, k2_x, w, h, x)
    y = solve_one(k1_y, k2_y, w, h, y)
    return x, y


def solve_one(k1, k2, w, h, v):
    rank = w * h
    geta = lambda i, j, s: k1(j) if i == j else (k2(j) if j in s else 0)
    getr = lambda i: [geta(i, j, siblings(i, w, h)) for j in xrange(rank)]
    getm = lambda: [getr(i) for i in xrange(rank)]
    A = np.array(getm(), np.float)
    B = np.array(-v, np.float)
    return np.linalg.solve(A, B)


if __name__ == '__main__':
    main()
