"""Auditable equations and feasible controls for the original fishing model.

NumPy/SciPy only. No changes to the original notebooks are required.
The maximum-Hamiltonian convention is H = p*f - L, p(T) = -phi_y.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import erf


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
    rng = np.random.RandomState(42)  # exactly the notebook's geometry
    grounds = rng.rand(7, 2) * 200
    ships = rng.rand(5, 2) * 200
    if harbor:
        ships[:] = [100., 0.]
    return Problem(grounds, ships, T=T)


def norm(a, axis=-1, keepdims=False):
    # Analytic extension permits complex-step derivative verification.
    return np.sqrt(np.sum(a * a, axis=axis, keepdims=keepdims))


def geometry(x, F, u, problem):
    q = problem.grounds[None] - x[:, None]
    rho = norm(q)
    if np.any(np.abs(rho) < 1e-14):
        raise ValueError('Original direction is undefined at a ground center.')
    d = q / rho[..., None]
    g = np.exp(-problem.alpha * rho**2)
    C = np.exp(-problem.sigma * rho**2)
    s = np.sum((u * g * F)[..., None] * d, axis=1)
    return q, rho, d, g, C, s


def rhs(y, u, v, problem):
    x, F = y[:2*problem.N].reshape(-1, 2), y[2*problem.N:]
    *_, C, s = geometry(x, F, u, problem)
    sn = norm(s)
    if np.any(sn == 0):
        raise ValueError('Zero attraction: original normalized field is undefined.')
    return np.r_[(v[:, None] * s / sn[:, None]).ravel(),
                 -problem.rate * F * C.sum(axis=0)]


def stable_rhs(y, u, v, problem):
    """Real-valued paper dynamics without an artificial attraction-speed floor.

    Scale positive attraction strengths in log space before summing directions.
    This cancels exactly in s/|s| and prevents Gaussian underflow from stopping
    distant boats. Stopped boats have x'=0; a moving boat with undefined heading
    is rejected. Working schedules stay away from every ground center.
    ``rhs`` above retains the analytic extension used for derivative tests.
    """
    x, F = y[:2*problem.N].reshape(-1, 2), y[2*problem.N:]
    if np.any(u < 0) or np.any(u > 10) or np.any(v < 0) or np.any(v > problem.V+1e-12):
        raise ValueError('Attraction or speed outside the original control bounds.')
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


def hamiltonian(y, p, u, v, problem):
    return p @ rhs(y, u, v, problem) - running_cost(y, v, problem)


def adjoint(y, p, u, v, problem):
    """Correct continuous adjoint, holding controls fixed in partial derivatives."""
    N = problem.N
    x, F = y[:2*N].reshape(-1, 2), y[2*N:]
    px, pf = p[:2*N].reshape(-1, 2), p[2*N:]
    q, rho, d, g, C, s = geometry(x, F, u, problem)
    sn = norm(s)
    n = s / sn[:, None]
    Q = np.eye(2)[None] - n[:, :, None] * n[:, None, :]
    Dxs = np.sum((u*g*F/rho)[..., None, None] * (
        (2*problem.alpha + 1/rho**2)[..., None, None]
        * q[..., :, None] * q[..., None, :] - np.eye(2)), axis=1)
    Qp = np.einsum('nij,nj->ni', Q, px)
    diff = x[None] - x[:, None]  # x_l - x_i
    crowd = 4*problem.beta * np.sum(diff / (
        problem.epsilon + np.sum(diff**2, axis=-1))[..., None]**2, axis=1)
    fish = 2*problem.rate*problem.sigma * np.sum(
        (pf*F*C)[..., None] * q, axis=1)
    movement = -(v/sn)[:, None] * np.einsum('nji,nj->ni', Dxs, Qp)
    pf_dot = problem.rate*pf*C.sum(axis=0) - np.sum(
        (v/sn)[:, None]*u*g*np.einsum('nkj,nj->nk', d, Qp), axis=0)
    return np.r_[(crowd + fish + movement).ravel(), pf_dot]


def control_gradient(y, p, u, v, problem):
    x, F = y[:2*problem.N].reshape(-1, 2), y[2*problem.N:]
    px = p[:2*problem.N].reshape(-1, 2)
    _, _, d, g, _, s = geometry(x, F, u, problem)
    sn = norm(s)
    n = s/sn[:, None]
    Qp = px - n * np.sum(n*px, axis=1)[:, None]
    Hu = (v/sn)[:, None]*g*F*np.einsum('nkj,nj->nk', d, Qp)
    Hv = np.sum(px*n, axis=1) - problem.gamma
    return Hu, Hv


def maximize_hamiltonian(y, p, problem, attraction_upper=10.):
    """Exact pointwise maximization in 2D for 0 <= u <= upper, 0 <= v <= V.

    With positive fish/vision, reachable headings form the cone of ground rays.
    A cone direction in R^2 needs at most two rays. If p lies outside the cone,
    the best heading is an extreme ray; test all individual ground directions.
    This is a PMP update, not a guarantee that a full sweep decreases J.
    """
    if attraction_upper <= 0:
        raise ValueError('Attraction upper bound must be positive.')
    x,F = y[:2*problem.N].reshape(-1,2),y[2*problem.N:]
    if np.any(F <= 0):
        raise ValueError('Cone reconstruction requires positive fish amounts.')
    px = p[:2*problem.N].reshape(-1,2)
    q = problem.grounds[None]-x[:,None]
    rho = np.linalg.norm(q,axis=2)
    if np.any(rho == 0):
        raise ValueError('Original direction is undefined at a ground center.')
    d = q/rho[...,None]
    log_strength = -problem.alpha*rho**2 + np.log(F)[None]
    controls = np.zeros((problem.N,problem.K))
    speeds = np.zeros(problem.N)
    for i in range(problem.N):
        best = int(np.argmax(d[i]@px[i]))
        active = [best]
        coeff = np.ones(1)
        heading = d[i,best]
        pn = np.linalg.norm(px[i])
        if pn > 0:
            desired = px[i]/pn
            found = False
            for a in range(problem.K):
                for b in range(a):
                    matrix = d[i,[a,b]].T
                    if abs(np.linalg.det(matrix)) < 1e-12:
                        continue
                    c = np.linalg.solve(matrix,desired)
                    if np.all(c >= 0):
                        positive = c > 0
                        active = np.array([a,b])[positive]
                        coeff = c[positive]
                        heading = desired
                        found = True
                        break
                if found:
                    break
        log_u = np.log(coeff)-log_strength[i,active]
        controls[i,active] = attraction_upper*np.exp(log_u-log_u.max())
        speeds[i] = problem.V if px[i]@heading-problem.gamma > 0 else 0.
    return controls,speeds


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


def gaussian_exposure(segment, grounds, sigma):
    """Exact integral of exp(-sigma*distance^2) over a straight segment."""
    L = segment.end-segment.start
    a = segment.x-grounds
    speed = np.linalg.norm(segment.velocity)
    if speed == 0:
        return L*np.exp(-sigma*np.sum(a*a, axis=1))
    direction = segment.velocity/speed
    parallel = a @ direction
    perpendicular = a[:, 0]*direction[1]-a[:, 1]*direction[0]
    return np.exp(-sigma*perpendicular**2)*np.sqrt(np.pi)/(2*speed*np.sqrt(sigma)) * (
        erf(np.sqrt(sigma)*(parallel+speed*L))-erf(np.sqrt(sigma)*parallel))


def reciprocal_distance_integral(a, b, duration, epsilon):
    """Exact integral of 1/(epsilon + |a+b*t|^2), including crossings."""
    A = b @ b
    c = epsilon + a @ a
    if A < 1e-28:
        return duration/c
    B = a @ b
    determinant = A*epsilon + (a[0]*b[1]-a[1]*b[0])**2
    root = np.sqrt(determinant)
    angle = np.arctan2(A*duration*root, determinant+B*(B+A*duration))
    return angle/root


def pair_crowding(first, second, epsilon):
    total = 0.
    for a in first:
        for b in second:
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if hi > lo:
                total += reciprocal_distance_integral(
                    a.position(lo)-b.position(lo), a.velocity-b.velocity, hi-lo, epsilon)
    return total


def evaluate_routes(problem, routes, dwells, standoff=.5):
    trajectories = [route_segments(problem, i, route, dwell, standoff)
                    for i, (route, dwell) in enumerate(zip(routes, dwells))]
    return evaluate_trajectories(problem, trajectories), trajectories


def return_segments(problem, ship, targets, fraction, standoff=.5):
    """Explicit model extension: harbor target -1 and x(T)=home.

    Travel at V, divide available fishing time between up to two stops, return
    exactly at T. The new scalar fraction allocates dwell time, not new dynamics.
    Return None if even zero dwell cannot satisfy the speed/time constraints.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('Dwell fraction must be in [0,1].')
    home=problem.ships[ship]
    points=[home.copy()]
    for target in targets:
        q=problem.grounds[target]-points[-1]
        distance=np.linalg.norm(q)
        if distance <= standoff:
            points.append(points[-1].copy())
        else:
            points.append(points[-1]+q*(1-standoff/distance))
    points.append(home.copy())
    times=[np.linalg.norm(b-a)/problem.V for a,b in zip(points[:-1],points[1:])]
    slack=problem.T-sum(times)
    if slack < -1e-10:
        return None
    slack=max(0.,slack)
    dwells=[slack] if len(targets)==1 else [fraction*slack,(1-fraction)*slack]
    out=[]
    t=0.
    for k,target in enumerate(list(targets)+[-1]):
        travel=times[k]
        if travel>0:
            end=problem.T if target==-1 else t+travel
            velocity=(points[k+1]-points[k])/(end-t)
            out.append(Segment(t,end,points[k].copy(),velocity,target))
            t=end
        if target!=-1 and dwells[k]>0:
            out.append(Segment(t,t+dwells[k],points[k+1].copy(),np.zeros(2),target))
            t+=dwells[k]
    return out


def evaluate_return_routes(problem, routes, fractions, standoff=.5):
    trajectories=[return_segments(problem,i,route,fraction,standoff)
                  for i,(route,fraction) in enumerate(zip(routes,fractions))]
    if any(tr is None for tr in trajectories):
        return {'J_reduced':float('inf')}, []
    return evaluate_trajectories(problem,trajectories),trajectories


def evaluate_trajectories(problem, trajectories):
    exposure = sum((gaussian_exposure(s, problem.grounds, problem.sigma)
                    for trajectory in trajectories for s in trajectory), np.zeros(problem.K))
    F = problem.F0*np.exp(-problem.rate*exposure)
    length = sum((s.end-s.start)*np.linalg.norm(s.velocity)
                 for trajectory in trajectories for s in trajectory)
    crowd = 2*problem.beta*sum(pair_crowding(trajectories[i], trajectories[j], problem.epsilon)
                              for i in range(problem.N) for j in range(i))
    travel_cost = problem.gamma*length
    fish_cost = problem.delta*F.sum()
    return dict(J_reduced=crowd+travel_cost+fish_cost,
                J_full=crowd+travel_cost+fish_cost+problem.self_cost,
                crowding=crowd, travel=travel_cost, terminal_fish=fish_cost,
                harvested=problem.K*problem.F0-F.sum(), remaining=F.tolist(),
                distance=length)
