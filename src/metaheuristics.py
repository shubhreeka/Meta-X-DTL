"""Population-based metaheuristic optimisers, implemented from their original papers.

All optimisers MINIMISE `fitness(x)` over a box [lower, upper] and return
(best_x, best_cost, history) where history is the best cost after each iteration.

- PSO    : Kennedy & Eberhart (1995), inertia-weight variant (Shi & Eberhart, 1998)
- GWO    : Mirjalili, Mirjalili & Lewis (2014), Grey Wolf Optimizer
- FA     : Yang (2008), Firefly Algorithm
- GA-PSO : hybrid of a real-coded Genetic Algorithm and PSO (Kao & Zahara, 2008 style):
           each generation the better half is refined by PSO velocity updates and the
           worse half is replaced by GA offspring (tournament selection, blend crossover,
           Gaussian mutation).
"""
import numpy as np


def _init(n, lower, upper, rng):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return lower, upper, rng.uniform(lower, upper, size=(n, len(lower)))


def _evaluate(fitness, pop):
    return np.array([fitness(x) for x in pop])


# ---------------------------------------------------------------- PSO
def pso(fitness, lower, upper, n_particles=20, iters=30, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, seed=0):
    rng = np.random.default_rng(seed)
    lower, upper, x = _init(n_particles, lower, upper, rng)
    vmax = 0.2 * (upper - lower)
    v = rng.uniform(-vmax, vmax, size=x.shape)
    cost = _evaluate(fitness, x)
    pbest, pbest_cost = x.copy(), cost.copy()
    g = np.argmin(cost)
    gbest, gbest_cost = x[g].copy(), cost[g]
    history = []
    for t in range(iters):
        w = w_max - (w_max - w_min) * t / max(1, iters - 1)
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lower, upper)
        cost = _evaluate(fitness, x)
        improved = cost < pbest_cost
        pbest[improved], pbest_cost[improved] = x[improved], cost[improved]
        g = np.argmin(pbest_cost)
        if pbest_cost[g] < gbest_cost:
            gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]
        history.append(float(gbest_cost))
    return gbest, float(gbest_cost), history


# ---------------------------------------------------------------- GWO
def gwo(fitness, lower, upper, n_wolves=20, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    lower, upper, x = _init(n_wolves, lower, upper, rng)
    cost = _evaluate(fitness, x)
    history = []
    order = np.argsort(cost)
    alpha, beta, delta = x[order[0]].copy(), x[order[1]].copy(), x[order[2]].copy()
    a_cost, b_cost, d_cost = cost[order[0]], cost[order[1]], cost[order[2]]
    for t in range(iters):
        a = 2.0 - 2.0 * t / iters  # decreases linearly from 2 to 0
        new = np.empty_like(x)
        for i in range(n_wolves):
            pos = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(x.shape[1]), rng.random(x.shape[1])
                A, C = 2 * a * r1 - a, 2 * r2
                D = np.abs(C * leader - x[i])
                pos.append(leader - A * D)
            new[i] = np.mean(pos, axis=0)
        x = np.clip(new, lower, upper)
        cost = _evaluate(fitness, x)
        for i in np.argsort(cost):  # update the three leaders
            c = cost[i]
            if c < a_cost:
                delta, d_cost, beta, b_cost = beta, b_cost, alpha, a_cost
                alpha, a_cost = x[i].copy(), c
            elif c < b_cost:
                delta, d_cost = beta, b_cost
                beta, b_cost = x[i].copy(), c
            elif c < d_cost:
                delta, d_cost = x[i].copy(), c
        history.append(float(a_cost))
    return alpha, float(a_cost), history


# ---------------------------------------------------------------- Firefly
def firefly(fitness, lower, upper, n_fireflies=15, iters=25, beta0=1.0, gamma=1.0, alpha0=0.25,
            alpha_decay=0.97, seed=0):
    rng = np.random.default_rng(seed)
    lower, upper, x = _init(n_fireflies, lower, upper, rng)
    scale = upper - lower
    cost = _evaluate(fitness, x)
    alpha = alpha0
    history = []
    for _ in range(iters):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if cost[j] < cost[i]:  # j is brighter -> i moves towards j
                    r2 = np.sum(((x[i] - x[j]) / scale) ** 2)  # distance in normalised space
                    beta = beta0 * np.exp(-gamma * r2)
                    step = alpha * (rng.random(x.shape[1]) - 0.5) * scale
                    x[i] = np.clip(x[i] + beta * (x[j] - x[i]) + step, lower, upper)
                    cost[i] = fitness(x[i])
        alpha *= alpha_decay
        history.append(float(cost.min()))
    b = np.argmin(cost)
    return x[b].copy(), float(cost[b]), history


# ---------------------------------------------------------------- Hybrid GA-PSO
def ga_pso(fitness, lower, upper, pop_size=20, iters=30, crossover_rate=0.9, mutation_rate=0.1,
           w=0.7, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)
    lower, upper, x = _init(pop_size, lower, upper, rng)
    vmax = 0.2 * (upper - lower)
    v = np.zeros_like(x)
    cost = _evaluate(fitness, x)
    pbest, pbest_cost = x.copy(), cost.copy()
    history = []
    half = pop_size // 2
    for _ in range(iters):
        order = np.argsort(cost)
        x, v, cost = x[order], v[order], cost[order]
        pbest, pbest_cost = pbest[order], pbest_cost[order]
        gbest = pbest[np.argmin(pbest_cost)]

        # Elite half: PSO refinement
        r1, r2 = rng.random((half, x.shape[1])), rng.random((half, x.shape[1]))
        v[:half] = np.clip(w * v[:half] + c1 * r1 * (pbest[:half] - x[:half]) + c2 * r2 * (gbest - x[:half]), -vmax, vmax)
        x[:half] = np.clip(x[:half] + v[:half], lower, upper)

        # Weak half: replaced by GA offspring bred from the elite
        children = []
        while len(children) < pop_size - half:
            p1 = x[min(rng.integers(0, half, 2))]  # tournament of 2 among elites
            p2 = x[min(rng.integers(0, half, 2))]
            if rng.random() < crossover_rate:  # BLX-0.5 blend crossover
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                span = hi - lo
                child = rng.uniform(lo - 0.5 * span, hi + 0.5 * span)
            else:
                child = p1.copy()
            mut = rng.random(len(child)) < mutation_rate
            child[mut] += rng.normal(0, 0.1, mut.sum()) * (upper - lower)[mut]
            children.append(np.clip(child, lower, upper))
        x[half:] = np.array(children)
        v[half:] = 0

        cost = _evaluate(fitness, x)
        improved = cost < pbest_cost
        pbest[improved], pbest_cost[improved] = x[improved], cost[improved]
        history.append(float(pbest_cost.min()))
    b = np.argmin(pbest_cost)
    return pbest[b].copy(), float(pbest_cost[b]), history


OPTIMISERS = {"PSO": pso, "GWO": gwo, "FA": firefly, "GA-PSO": ga_pso}
