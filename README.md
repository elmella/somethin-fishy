## Somethin' Fishy - Fishing and Optimal Control

**Pontryagin repair (October 2026):** The forward–backward solver now uses forward states, backward costates, and joint Hamiltonian control updates. The route/timing solver is retained as a benchmark.

On the original 400-unit notebook map, the best verified PMP policy harvests **6.9973 / 7** and reduces the benchmark's original cost from **12.1542 to 9.6056** (about **21%**). A separate run from an unoptimized coverage guess also passes the regularized numerical convergence checks; it uses no route-optimizer solution.

**The mathematical changes are explicit:** effective-target feedback coordinates, a 0.5-unit target regularization, and a 0.001-speed travel-cost smoothing. The latter changes the cost by at most 0.02 over this horizon; comparisons report the original cost. Original depletion and physical movement equations, weights, and control bounds are preserved in the tested feedback replay. **Recompute attractions from the current state; a frozen raw-attraction table remains numerically fragile.** Convergence is claimed only for the regularized PMP formulation, not the unmodified raw-open-loop problem or a global optimum.

[Audit, derivation, numerical checks, and limitations](audit/REPORT.md) ·
[Side-by-side animation](audit/results/pontryagin/comparison.gif) ·
[Outcome plots](audit/results/pontryagin/outcomes.png) ·
[Best verified result](audit/results/pontryagin/best_400.json) ·
[Coverage-initialized result](audit/results/pontryagin/coverage_400_verified.json) ·
[600-unit result](audit/results/pontryagin/coverage_600_verified.json)

**Watch the 300-second comparison:** [animated GIF on the original map](audit/results/pontryagin/scenarios/baseline_300.gif). To play, pause, or scrub all four policies together, open the [scenario gallery](audit/scenario_gallery.html) locally after cloning. Each row below uses identical starting conditions across all four policies; values are fish harvested by the deadline.

| Starting conditions (animation) | No optimal control | Original broken optimal control | Repaired, crowding on (β = 5) | Repaired, crowding off (β = 0) |
| --- | ---: | ---: | ---: | ---: |
| [5 boats, 7 grounds, dispersed](audit/results/pontryagin/scenarios/baseline_300.gif) | 3.779 | 3.803 | 6.908 | 6.909 |
| [5 boats, 10 grounds, dispersed](audit/results/pontryagin/scenarios/grounds_300.gif) | 4.817 | 5.355 | 9.825 | 9.830 |
| [8 boats, 12 grounds, dispersed](audit/results/pontryagin/scenarios/fleet_300.gif) | 7.952 | 7.914 | 11.889 | 11.901 |
| [5 boats, 7 grounds, spaced along a harbor line](audit/results/pontryagin/scenarios/harbor_300.gif) | 2.000 | 2.000 | 6.440 | 6.443 |

“No optimal control” holds attractions and commanded speeds at one. “Original broken optimal control” runs the historical notebook’s single control update and then simulates the updated controls; the notebook itself plotted the trajectory from before that update. The repaired columns optimize the same regularized system, changing only the crowding weight in the objective. See the [unrounded results and numerical checks](audit/results/pontryagin/scenarios/summary.json).

**What changed in the math?** The [before-and-after equations](audit/equation_comparison/main.tex) distinguish corrections to the original costates and speed rule from later modeling choices. The final solver chooses a steering target and throttle for each boat and reconstructs bounded attractions during replay. The [interactive movement explanation](audit/movement_explained.html) shows why the target gives the original instantaneous direction and how the new movement law slows the boat near it. Open the HTML locally to use the controls.

The 600-unit coverage-initialized run also passes the declared checks: original cost **13.0897** versus benchmark **13.5155**, with **6.9990 / 7** harvested. Its residuals are closer to the acceptance thresholds; the report gives the exact values. All **46 regression tests** and both notebooks' saved-result playback runs pass.

Run with `audit/requirements.txt` installed:

```sh
python audit/run_notebook.py
python audit/pontryagin_controller.py --seed coverage --name coverage_400
python audit/working_controller.py --scenario all  # retained benchmark
python audit/scenario_experiments.py  # 300-second cases; numerical solves can be slow
python audit/scenario_gallery.py      # four-policy animations and offline viewer
python -m unittest discover -s audit -v
```

The notebooks distinguish saved validated playback from recomputation. The reported PMP runs use the notebooks' original dispersed starts. The paper's shared-harbor route benchmark remains available; PMP optimization of that scenario is not yet validated. Final positions remain free, as in the supplied endpoint conditions.

The historical project write-up below is preserved. Its original optimality claims should not be read as validated conclusions.

    Sam Layton
    Xander de la Bruere
    Jakob Gertsch
    Matt Mella
    Wilson Stoddard

### Abstract

We investigate the use of optimal control to maximize the fish harvested by a large fleet of fishing ships. In particular, we seek to use Pontryagin’s Maximum Principle and bang-bang control theory to find the most efficient path for each ship to take given that we can control the ships’ targets.

<div align="center">
  <img src="visualizations/controlled-notcontrolled.gif" alt="This is a PNG image." width=900>
</div>
    

### Background

When developing a project idea, we wanted to focus on a broad class of problems involving the control of a group of agents moving through space in order to visit points of interest. To illustrate, controlling a fleet of drones photograph potential threats on a battlefield, flying a group of helicopters to dump water on forest fires, and using drones to do do search-and-rescue are all examples of this type of problem.

This project seeks to apply the classical principles and theory of optimal control to the age-old problem of fishing. Our hope is that the results of our work can be generalized to solve optimal control problems involving the coordination of agents visiting points of interest.

Optimizing fishing harvest is of great interest for several reasons. First, the fishing industry is a significant contributor to GDP throughout the world, as evidenced by its estimated global market value of \$277 billion in 2020 [1]. Since the value that the fishing industry adds to the economy is directly related to the amount of fish caught and sold, optimizing fishing harvests is an important problem to solve for those seeking to boost national economies, not to mention the benefits to the commercial fishermen who actually work in the industry.

Second, more optimal fishing would allow more people greater access to the nutritional benefits of seafood consumption. A 2022 study found that 90\% of Americans do not consume seafood at levels recommended by the US Dietary Guidelines for Americans, and that low-income individuals cite high seafood prices as the primary barrier to higher seafood consumption [2]. Finding a way to harvest more fish using the same amount of labor and capital could increase the supply of various types of seafood, driving down prices and increasing access to the health benefits of seafood for low-income individuals.

Finally, optimized fishing can help relieve problems caused by invasive species that plague lakes and rivers throughout the world. For example, many lakes and rivers in the Southeast of the United States still suffer from invasive silver and bighead carp that entered these waters during extensive flooding in the 1970s [3]. The presence of these invasive aquatic species continues to compromise the water quality and aquatic biodiversity of these waters today. Learning to effectively harvest and dispose of these fish would thus ease or perhaps completely remove the strain on these important ecosystems.

Notwithstanding the impact of optimized fishing on economic well-being, health, and aquatic ecosystems  we were surprised to find that no research has focused on the problem of optimizing group fishing efforts, leaving commercial fishermen to solve this problem on their own. While research exists for optimizing fish harvest in fisheries [4], and while other research has investigated the role of optimal control in maneuvering at sea [5], no extant research attempts addresses the issue of optimizing fishing harvest at sea. This analysis seeks to fill this gap in the literature by applying optimal control theory to maximizing the harvest of a group of fishing boats at sea.
### Problem Description

Our objective is to maximize the total amount of fish harvested by a fleet of ships within a given time interval, which we refer to as the optimal fishing problem.

We operate under the assumption that fish are distributed in various fishing grounds across the sea, with each location featuring a normal distribution of fish populations. Additionally, we assume that fishing ships are more likely to locate these grounds the closer they are to them. Therefore, we simplify our model by assuming that fishing can only occur in specific areas of the sea, which we designate as fishing grounds.

Given this assumption, rather than following the classical approach of controlling ship acceleration directly, we test a novel approach to our optimal control method. We instead control which fishing grounds each ship should target, as if we were orchistrating a sophisticated network of attractive forces between ships and fishing grounds. This method ultimately causes each ship to set sail towards a fishing ground that, while not necessarily closest, optimizes the collective harvest among all ships. 

Specifically, we first control the speed of each boat in the fleet. We require this control variable to be within an admissible control region that has zero as a lower bound and a tunable maximum speed parameter as an upper bound. We then channel the speed of each boat by tracking and controlling a vector of attractions between each ship and each fishing ground. Together, these attractions additively determine the ship’s overall direction, as demonstrated in the figure below.

<div style="text-align: center;">
    <img src="visualizations/boat_trajectory.png" alt="This is a PNG image." width=400>
</div>

Thus, the optimal fishing problem we solve is finding (1) the speed control and (2) attraction network that maximizes the collective amount of fish harvested at each step before the time expires.
### Mathematical Representation

In order to use Pontryagin's Maximum Principle (PMP) to determine the optimal speed control and attraction network evolution, we construct a customized state-space model to provide mathematical structure for this problem.

We model the ocean as a two-dimensional plane, with the origin at the bottom left corner. We assume there are $N$ fishing ships and $K$ fishing grounds. To model the dynamics of the fishing environment, we incorporate into our state space the horizontal and vertical position of ship $i$, $x_i \in \mathbb{R}^2$. In addition, the amount of fish remaining in each fishing ground $j$, represented by $F_{j}$, is scaled as a percentage relative to full capacity. Each fishing ground is represented as a point corresponding to its center, $f$. This setup is visualized in the diagram below.

<div style="text-align: center;">
    <img src="visualizations/fishing_environment.png" alt="This is a PNG image." width=700>
</div>

In modeling a ship's relative attraction to one fishing ground over another, we want fishing grounds closest to a given ship to have the strongest relative attraction, while fishing grounds far away from a ship have essentially zero attraction. To accomplish this, we first define a scalar term to represent Gaussian vision from each ship $i$ to each fishing ground $j$ as $g_{ij} = e^{-\alpha \lVert f_j - x_i \rVert^2}$. The parameter $\alpha$ governs how quickly a fishing ground disappears out of sight as a function of its distance from the ship.

We then define the unit vector $d_{ij} = \frac{f_j - x_i}{\lVert f_j - x_i \rVert}$, which points in the direction from ship $i$ to fishing ground $j$. Having done so, we now define the attraction force of fishing ground $j$ to ship $i$ as the vector $A_{ij} = u_{ij}g_{ij}d_{ij}F_j$, where $u_{ij}$ is a control variable used to tune the natural attraction force. Note that the natural attraction force increases with $F_j$, the current amount of fish remaining at ground $j$, and Gaussian vision $g_{ij}$. Finally, we define the net force of attraction on ship $i$, the sum of all fishing ground attraction forces, as 

```math
s_i = \sum_{j=1}^{K}A_{ij}.
```

Having set up the model, we now define our first-order state equations  $\dot{y} \in \mathbb{R}^{2N+K}$ as a concatenation of $\dot{x_i}$ and $\dot{F_j}$:
```math
\begin{align}
\dot{x_i} &= u_{i,k+1}\frac{s_i}{\lVert s_i \rVert} \in \large \mathbb{R}^2 \notag \\
\dot{F_j} &= -rF_j\sum_{i=1}^{N}e^{-\sigma \lVert f_j - x_i \rVert^2} \in \large \mathbb{R}. \notag
\end{align}
```
Notice that vector $\dot{y}^{\top}$ contains the change in each boat's $x$- and $y$-position coordinates as well as the amount fish remaining at each fishing ground:
```math
\begin{equation}
\dot{y}^{\top} = \left[\dot{x}_{11}, \dot{x}_{12}, \dot{x}_{21}, \dot{x}_{22}, \ldots, \dot{x}_{N1}, \dot{x}_{N2}, \dot{F}_1, \dot{F}_2, \ldots, \dot{F}_K \right]. \notag
\end{equation}
```
We assume that all ships start from a harbor $h \in \mathbb{R}^2$ and that all fishing grounds start at full capacity. Thus we require initial conditions $x_i(0) = h$ and $F_j(0) = F_0$, while $x_i(t_f)$ and $F_j(t_f)$ are free.

Given the state evolution we defined, we now seek to find a function for our control that maximizes the fishing harvest. To do this, we define our cost functional:

```math
\begin{equation}
J[u] = \int_{0}^{tf} \left(\beta \sum_{i=1}^{N}\sum_{l=1}^{N} \frac{1}{\varepsilon +\lVert x_l - x_i \rVert^2 } + \gamma \sum_{i=1}^{N} \lVert \dot{x_i} \rVert\right)dt + \delta\sum_{j=1}^{K}F_j(t_f). \notag
\end{equation}
```

The cost functional consists of three main terms, each scaled by a tunable parameter. The first term 

```math 
\sum_{i=1}^{N}\sum_{l=1}^{N} \frac{1}{\varepsilon +\lVert x_l - x_i \rVert^2 }
```

is large when ships are close together, thus rewarding boats for spreading out rather than converging to the same fishing ground. The second

```math
\gamma \sum_{i=1}^{N} \lVert \dot{x_i} \rVert
```

penalizes curved paths, which are costly in terms of both time and fuel. The last term 
 
```math 
\delta\sum_{j=1}^{K}F_j(t_f)
```

penalizes the amount of fish remaining unharvested in the fishing grounds at the end of the time interval.

We construct co-state equation $p_i \in \mathbb{R}^2$, which corresponds to $x_i$. In addition, we define $p_{F_j}$, which corresponds to $F_j$. Both co-state equations are examined more closely in the following section. Looking at endpoint conditions, $y$ is free so we let $p_i(t_f) = 0$. Due to our functional taking the Bolza form, we construct our endpoint condition by taking the derivative with respect to $y$, yielding $p_{F_j}(t_f) = -\delta$. Together, these modeling choices lead to the Hamiltonian:

```math
\begin{equation}
H = \sum_{i=1}^{N}p_i \cdot \dot{x_i} + \sum_{j=1}^{K}p_{F_j}\dot{F}_j - \beta \sum_{i=1}^{N}\sum_{l=1}^{N} \frac{1}{\varepsilon +\lVert x_l - x_i \rVert^2 } - \gamma \sum_{i=1}^{N} \lVert \dot{x_i} \rVert. \notag
\end{equation}
```

In summary, the table below provides a succinct description of each parameter in our model.

| Parameter | Description | Parameter | Description |
| --- | --- | ---| --- |
| $F_0 \in \mathbb{R}$ | Starting quantity of fish in water | $r \in \mathbb{R}$ | Rate of fish collection |
| $N \in \mathbb{R}$ | Number of ships in the model |  $K \in \mathbb{R}$ | Number of fishing grounds in the model |
| $V \in \mathbb{R}$ | Maximum speed for ships | $\alpha \in \mathbb{R}$ | Vision of ships |
| $\sigma \in \mathbb{R}$ | Inverse size of fishing ground | $h \in \mathbb{R}^{2}$ | Harbor location |
| $t_0 \in \mathbb{R}$ | Initial time | $f_j \in \mathbb{R}^{2}$ | Location of fishing ground $j$ |
| $t_f \in \mathbb{R}$ | Final time | $\beta \in \mathbb{R}$ | Crowding penalty |
| $\gamma \in \mathbb{R}$ | Distance penalty | $\delta \in \mathbb{R}$ | Unharvested fish penalty |

The following table indicates the variables of the model.

| Variable | Description |
|---|---|
| $x_i \in \mathbb{R}^{2}$ | Ship location |
| $u_{ij} \in \mathbb{R}$ | Control on ship $i$ for fishing ground $j$ |
| $u_{i,k+1} \in \mathbb{R}$ | Speed control for ship $i$ |
| $F_j \in \mathbb{R}$ | Quantity of fish at ground $j$ |
### Solution

##### Derivation of Pontryagin Maximizing Principle

Given our Hamiltonian, we can now derive our optimal control for both $u_{ij}$ (the toggle on the fishing ground attraction) and $u_{i,k+1}$ (the speed control of ship $i$). We take the derivative of our Hamilton with respect to $u_{ij}$ and set $D_{u_{ij}}H = 0$. To calculate $H_{u_{ij}}$, note that all terms are constant in terms of $u_{ij}$ except the first, as $\dot{x_i}$ is the only one depending on $u_{ij}$. Thus, all terms will vanish except $p_i \cdot \dot{x_i}$. We note that $\lVert \dot{x_i} \rVert = u_{i,k+1}$ when substituting in for $\dot{x_i}$, hence it does not contribute to our derivative of $u_{ij}$. Therefore, $D_{u_{ij}}H = D_{u_{ij}}p_i \cdot \dot{x_i} = p_i \cdot D_{u_{ij}} \dot{x_i} = 0$. Substituting in for $\dot{x_i}$, we obtain

```math
\begin{equation}
p_i \cdot D_{u_{ij}} u_{i,k+1}\frac{s_i}{\lVert s_i \rVert} = u_{i,k+1}p_i \cdot D_{u_{ij}}\frac{s_i}{\lVert s_i \rVert} = 0. 
\end{equation}
```
Taking the derivative of the unit vector $s_i$ is a tricky process, involving differentiation rules of multivariate calculus, the quotient rule, and the chain rule. We invoke the following lemma and leave the proof as an exercise to the reader. 

**Lemma 1**
Let function $\textbf{s}: \mathbb{R}^n \to \mathbb{R}^m$ where $\textbf{s}(\textbf{y})$ is a function of variable $\textbf{y}$. Taking the derivative of $D_{y}\frac{\textbf{s}}{\lVert \textbf{s}\rVert}$ of the unit vector is given by:

```math
D_{y}\frac{\textbf{s}}{\lVert \textbf{s}\rVert}=\frac{1}{\lVert \textbf{s} \rVert}\left(\textbf{I} - \frac{\textbf{s}\textbf{s}^T}{\lVert \textbf{s} \rVert ^2}\right)D_y\textbf{s}.$$
```

Note that $\frac{\textbf{s}\textbf{s}^T}{\lVert \textbf{s} \rVert ^2}$ is the outer product of $\textbf{s}$ divided by its own inner product. This leads us to our second lemma, with its proof also left as an exercise for the reader.

**Lemma 2**
Given vector $\textbf{s} \in \mathbb{R}^n$, the matrix $\frac{\textbf{s}\textbf{s}^T}{\lVert s \rVert ^2}$ represents a linear transformation equivalent to the projection of vector $\textbf{x} \in \mathbb{R}^n$ onto vector $\textbf{s}$. That is,

```math
\frac{\textbf{s}\textbf{s}^T}{\lVert \textbf{s} \rVert ^2}\textbf{x} = \text{proj}_{\textbf{s}}\textbf{x}.
```

Additionally, taking $\textbf{I} - \frac{\textbf{s}\textbf{s}^T}{\lVert \textbf{s} \rVert ^2}$ corresponds to the linear transformation giving the orthogonal complement of vector $\textbf{x}$ projected onto $\textbf{s}$, i.e. $\textbf{x} - \text{proj}_{\textbf{s}}\textbf{x}$. 

The following is an illustration of the concept of Lemma 2
<div style="text-align: center;">
    <img src="visualizations/lemma2.png" alt="This is a PNG image." width=400>
</div>


Returning to our optimal control, we invoke $\mathbf{Lemma 1}$ to take the derivative

```math
\begin{equation}
u_{i,k+1}p_i \cdot D_{u_{ij}}\frac{s_i}{\lVert s_i \rVert} = u_{i,k+1}p_i \cdot \frac{1}{\lVert s_i \rVert}\left(I - \frac{s_is_i^T}{\lVert s_i \rVert ^2}\right)D_{u_{ij}}s_i = 0.
\end{equation}
```

To complete the derivative, we take $D_{u_{ij}}s_i = g_{ij}F_jd_{ij}$. Note that $\lVert s_i \rVert$, $g_{ij}$, $u_{i,k+1}$, and $F_j$ are all scalars, hence if $D_{u_{ij}}H = 0$, we can eliminate these terms such that

```math
\begin{align}
    p_i \cdot \left(I - \frac{s_is_i^T}{\lVert s_i \rVert ^2}\right)d_{ij} = 0.
\end{align}
```

Invoking $\textbf{Lemma 2}$, we simplify the above condition to be

```math
\begin{align}
    p_i^{T}(d_{ij} - \text{proj}_{s_i}d_{ij}) = 0.
\end{align}
```

Satisfying this condition will thus give the optimal control. Upon further inspection, we note that the orthogonal component of $d_{ij}$ projected onto $s_i$ must be orthogonal to $p_i$. Because our state space exists in $\mathbb{R}^2$, an orthogonal vector of an orthogonal vector exists in the same space as the original vector. Thus the optimal $u_{ij}$ for each ship-fishing ground pair will attempt to align both $p_i$ and $s_i$ for each ship. 

Additionally, taking $D_{u_{i,k+1}}H = 0$ implies a bang-bang solution for speed where:

```math
u_{i,k+1} =
\begin{cases}
V & \text{if } \frac{p_i^{T}s_i}{\lVert s_i \rVert} + \gamma > 0,\\
0 & \text{else}
\end{cases}
```

To find our co-state equations, we take both $\dot{p_i} = -D_{x_i}H$ corresponding to each ship in our state space and $\dot{p_{F_j}} = -D_{F_j}H$ corresponding to each fishing ground. Looking first at the co-state for each boat, we begin by defining the derivative of $s_i$ with respect to $x_i$:

```math
\begin{equation}
D_xs_i = \sum_{j=1}^{K}\frac{u_{ij}g_{ij}F_j}{\lVert f_j - x_i \rVert}\left(\left(\alpha + \frac{1}{\lVert f_j - x_i \rVert^2}\right)(f_j - x_i)(f_j - x_i)^{T} - I\right).
\end{equation}
```

We substitute the above expression as we once again invoke $\textbf{Lemma 1}$ and take the derivative of our Hamiltonian. We arrive at

```math
\begin{align}
\dot{p_i} &= 4\beta\sum_{l=1}^{N}\frac{x_l - x_i}{\left(\varepsilon +\lVert x_l - x_i \rVert^2 \right)^2}
- 2r\sigma\sum_{j=1}^{K}\left(p_{F_j}e^{-\sigma \lVert f_j - x_i \rVert^2}\left(f_j - x_i\right) \right)
- u_{i,k+1}\left(I - \frac{s_is_i^T}{\lVert s_i \rVert ^2}\right)\frac{D_xs_i}{\lVert s_i \rVert}p_i
\end{align}
```

```math
\begin{align}
\dot{p}_{F_j} &= rp_{F_j}\sum_{i=1}^{N}e^{-\sigma \lVert f_j - x_i \rVert^2} %- \sum_{i=1}^{N}\frac{u_{ij}u_{i,K+1}g_{ij}}{\lVert s_i \rVert}p_i^T\left(I - \frac{s_is_i^T}{\lVert s_i \rVert ^2}\right)d_{ij}%
\end{align}
```

Our state equations, maximizing conditions, bang-bang speed control, and co-state equations are all thus defined, ready to be solved numerically to compute our optimal solution.

##### Implementation

Solving this first-order system of differential equations is no easy task. Not only are the evolution equations complicated, but the problem involves a boundary value problem with bang-bang controls for speed, as well as continuous controls for each fishing ground. Furthermore, our continuous controls are implicitly defined and cannot practically be derived analytically. Hence, we tackle these obstacles in implementation one at a time.

First, we address the boundary value problem with bang-bang controls by implementing a method similar to the one found in the Timber Harvesting lab. We begin with an initial guess for our controls, solve our state forward, then solve our co-state backward using RK4. We then solve for our controls at each time step. This process is repeated until our control solution converges.

Solving the implicit controls $u_{ij}$ at each step was the second hurdle we overcame in implementation. As previously shown in our derivation of PMP conditions, we want to find the correct toggle of controls for each fishing ground such that we align $p_i$ and $s_i$ in the same subspace. Thus, we find our control toggles for each time step by defining a minimization problem that penalizes these vectors being out of alignment. Essentially, we take the vector orthogonal to $s_i$ and its inner product with $p_i$ then solve the minimization problem
```math
\min_{u_{i1}...u_{ik}} \quad p_i \cdot \text{ortho}(s_i) + \lambda R_{it}.
```
We introduce $R_{it}$ as a regularization term that encourages continuity of control across time. We define $R_{it}$ by taking the previous timestep's control and letting 

```math
R_{it} = \sum_{j=1}^k (u_{ij,t-1} - u_{ij})^2
```

In the figure below, we can see that including $R_{it}$ in the control leads a much more continuous solution than solutions found without it.

<div style="text-align: center;">
    <img src="visualizations/no_reg.png" alt="This is a PNG image." width=400>
</div>

<div style="text-align: center;">
    <img src="visualizations/with_reg.png" alt="This is a PNG image." width=400>
</div>

Thus, for each time step we iterate forward and backward, solving this minimization problem each time step to find the evolution of our optimal control. Once these implicitly-found controls converge after a sufficient number of iterations, we have our final control throughout time. See the figure below.  

The final caveat we must address is how our minimization problem involves $p_i$, which itself will change if $u_{ij}$ changes. Thus, rather than trying to pin down this moving target, we simply use the $p_i$ at the previous iteration so that our implementation is more tractable.

### Interpretation

After implementing our solution, we find that for small and simple systems, we are able to successfully control the trajectory of the ships in an optimal manner. In particular, we observe that ships crowd together less and travel in straighter paths. In addition, we find that after implementing our optimal control, the inner product of the costate and the orthogonal vector to the attractive force sum term, $s_i$, vanishes, as predicted by our model.

<div style="text-align: center;">
    <img src="visualizations/optim.png" alt="This is a PNG image." width=700>
</div>

In addition, we find that by varying the parameters $\beta$, $\gamma$, and $\delta$—each scaling the impact of the various terms of our cost functional—we are able to control the system's motion in our desired manner. For example, the following images demonstrate the effect of changing our parameter governing ship crowding, $\beta$, from no control first, to $\beta=0$, then finally to $\beta=1$. As $\beta$ increases, we see less crowding, as expected.

<div style="text-align: center;">
    <img src="visualizations/nocontrol.png" alt="This is a PNG image." width=500>
</div>

<div style="text-align: center;">
    <img src="visualizations/betazero.png" alt="This is a PNG image." width=500>
</div>

<div style="text-align: center;">
    <img src="visualizations/beta1.png" alt="This is a PNG image." width=500>
</div>

While we have seen much success in improving optimal fishing harvest from our solution, more testing and modifications should be performed to ensure that the optimal solution produced is closer to the globally optimal solution. Since our optimal control is found by minimizing an inner product of many tunable parameters, there are many combinations of attraction vectors and costate vectors that produce local minima, making it likely for our algorithms to converge to a local minimum instead of the global minimum.

For example, we note that while the cost functional we have explored here penalizes the proximity of ships, encouraging ships to spread out, the cost functional does not directly penalize the case where many ships have set sights on the same fishing ground from far away. Future iterations of this model could mitigate the probability of this situation by making the control variable matrix for ships and fishing grounds column stochastic. Doing so would cause all control attraction forces for a given fishing ground to sum to 1. Under this model, if most ships set sights on a given fishing ground, the overall attraction level would be forced to be small, while fishing grounds with few ships nearby would have attraction levels scaled up. Future tests could examine whether this model ensures a more even spread of ships to various fishing grounds throughout time.

Nonetheless, our model and implementation of optimal control has clearly led to increased harvest over the given time interval in our simulations. While more improvements can be made, we feel optimistic that our novel approach to the optimal fishing problem outlined here, if refined and explored further, can contribute greatly to the success of fishing companies. We hope this will lead to increased economic output, better access to nutritional seafood, and reduced negative effects of invasive species for all.
### References

[1] Jannik Lindner. “The most surprising fishing industry statistics and trends in 2024,” 2023.

[2] David Love et al. “Affordability influences nutritional quality of seafood consumption among income and race/ethnicity groups in the united states,” 2022.

[3] U.S. Fish and Wildlife Service. “Invasive carp in southeastern waters,” 2021.

[4] Roland Pulch, Bernd Kugelmann. “Robust optimal control of fishing in a three competing species model,” 2018.

[5] Miele et al. “Optimal control of a ship for collision avoidance maneuvers,” 1999.
