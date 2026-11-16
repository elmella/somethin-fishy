"""Fishing geometry, physical feedback dynamics, and coverage initialization."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    grounds: np.ndarray
    ships: np.ndarray
    T: float = 400.
    alpha: float = .005
    sigma: float = .05
    rate: float = .05
    beta: float = 5.
    gamma: float = .01
    delta: float = 100.
    epsilon: float = .0001
    V: float = 1.
    F0: float = 1.

    @property
    def N(self):
        return len(self.ships)

    @property
    def K(self):
        return len(self.grounds)

    @property
    def self_cost(self):
        return self.beta * self.N * self.T / self.epsilon


def scenario(harbor=False, T=400.):
    rng = np.random.RandomState(42)
    grounds = rng.rand(7, 2) * 200
    ships = rng.rand(5, 2) * 200
    if harbor:
        ships[:] = [100., 0.]
    return Problem(grounds, ships, T=T)


def stable_rhs(y, u, v, problem):
    """Real-valued paper dynamics without an artificial attraction-speed floor.

    Scale positive attraction strengths in log space before summing directions.
    This cancels exactly in s/|s| and prevents Gaussian underflow from stopping
    distant boats. Stopped boats have x'=0; a moving boat with undefined heading
    is rejected. Working schedules stay away from every ground center.
    """
    x, F = y[:2*problem.N].reshape(-1, 2), y[2*problem.N:]
    if np.any(u < 0) or np.any(u > 10) or np.any(v < 0) or np.any(v > problem.V+1e-12):
        raise ValueError('Attraction or speed outside the physical control bounds.')
    if np.any(F < 0) or not np.all(np.isfinite(y)):
        raise ValueError('Nonphysical stock or nonfinite state.')
    q = problem.grounds[None] - x[:, None]
    rho = np.linalg.norm(q, axis=-1)
    if np.any(rho == 0):
        raise ValueError('Ground-center direction is undefined; stop short of centers.')
    positive = (u > 0) & (F[None] > 0)
    log_strength = np.full(u.shape, -np.inf)
    log_strength[positive] = (np.log(np.where(u > 0, u, 1.))
                             + np.log(np.where(F > 0, F, 1.))[None]
                             - problem.alpha*rho**2)[positive]
    moving = v > 0
    peak = log_strength.max(axis=1)
    if np.any(moving & ~np.isfinite(peak)):
        raise ValueError('A moving boat has no positive attraction.')
    peak = np.where(np.isfinite(peak), peak, 0.)
    scaled = np.exp(log_strength-peak[:, None])
    direction = np.sum(scaled[..., None]*q/rho[..., None], axis=1)
    magnitude = np.linalg.norm(direction, axis=1)
    if np.any(moving & (magnitude == 0)):
        raise ValueError('A moving boat has cancelling attraction vectors.')
    velocity = np.zeros_like(x)
    velocity[moving] = v[moving, None]*direction[moving]/magnitude[moving, None]
    depletion = -problem.rate*F*np.exp(-problem.sigma*rho**2).sum(axis=0)
    return np.r_[velocity.ravel(), depletion]


def running_cost(y, v, problem, include_self=True):
    x = y[:2*problem.N].reshape(-1, 2)
    diff = x[None] - x[:, None]
    inv = 1 / (problem.epsilon + np.sum(diff*diff, axis=-1))
    if not include_self:
        inv = inv.copy()
        np.fill_diagonal(inv, 0)
    return problem.beta * inv.sum() + problem.gamma * v.sum()


@dataclass
class Segment:
    start: float
    end: float
    x: np.ndarray
    velocity: np.ndarray
    target: int

    def position(self, t):
        return self.x + (t-self.start)*self.velocity


def route_segments(problem, ship, targets, dwell, standoff=.5):
    """One or two targets; cruise at V, dwell, cruise, then stop.

    Select u_target=1, all other u=0. Gaussian vision and F cancel in s/|s|.
    Stop standoff units before each center to avoid undefined d at the center.
    This restricts the admissible controls; it does not enlarge them.
    """
    x = problem.ships[ship].copy()
    t = 0.
    out = []

    def append(duration, velocity, target):
        nonlocal x, t
        end = min(problem.T, t + max(0., duration))
        if end > t:
            out.append(Segment(t, end, x.copy(), velocity.copy(), int(target)))
            x = x + (end-t)*velocity
            t = end

    for k, target in enumerate(targets):
        q = problem.grounds[target]-x
        distance = np.linalg.norm(q)
        if distance < 1e-12:
            raise ValueError('Route starts at undefined ground center.')
        travel = max(0., distance-standoff)/problem.V
        append(travel, problem.V*q/distance, target)
        append(dwell if k == 0 and len(targets) > 1 else problem.T,
               np.zeros(2), target)
        if t >= problem.T:
            break
    return out
