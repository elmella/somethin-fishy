# Steering, fishing, and the objective

Boat $i$ is at $x_i(t)$, fishing ground $j$ is at the fixed position $f_j$, and $F_j(t)$ is the stock remaining there. A boat chooses a steering target $z_i$ inside the convex hull of the grounds and a throttle $a_i$ between zero and one. With speed limit $V=1$ and slowing distance $\eta=0.5$, its velocity is

$$
\dot{x}_i=a_i V\frac{z_i-x_i}{\sqrt{\|z_i-x_i\|^2+\eta^2}}.
$$

The target may lie between grounds. Moving toward it only earns catch to the extent that the boat physically passes near fish. A stationary boat can still fish. Each ground depletes at rate

$$
\dot F_j=-r F_j\sum_i C_{ij},\qquad
C_{ij}=\exp(-\sigma\|f_j-x_i\|^2),\quad r=0.05,\quad \sigma=0.05.
$$

For given trajectories, the remaining stock is its starting value times the exponential of minus the integrated depletion rate. More boats near a ground increase fishing intensity, but there is less fish to catch as a ground empties.

## How an unplanned boat picks a target

Once per simulated second it evaluates the weights below, using its current position and the current stocks. Every selectable attraction is one. Let $d_j=\|f_j-x_i\|$ and $\alpha=0.005$; then

$$
b_j=\frac{e^{-\alpha d_j^2}F_j}{d_j},\qquad
z_i=\sum_j\frac{b_j}{\sum_m b_m}f_j.
$$

If the boat is exactly at a ground center with positive stock, the limiting target is that center. The boat holds its target and full throttle until the next decision. The optimized fleet instead holds its planned targets and throttles between schedule boundaries. Both fleets share the speed bound, slowing distance, fish law, initial states, and deadline. The catch comparison measures the benefit relative to this *specific* unplanned policy.

## What the optimizer values

Write $v_i=\|\dot x_i\|$. The minimized cost is

$$
J=\int_0^T\left[
2\beta\sum_{i<k}\frac{1}{\varepsilon+\|x_k-x_i\|^2}
+\gamma\sum_i\big(\sqrt{v_i^2+\nu^2}-\nu\big)
\right]dt+\delta\sum_j F_j(T).
$$

Here $\varepsilon=10^{-4}$, $\gamma=0.01$, $\nu=0.001$, and $\delta=100$. The factor of two counts both orders of each distinct pair; self-pairs would add only a constant. Total catch equals starting stock minus ending stock. Plots report physical distance; the solver uses the smooth travel penalty above. In the policy files, `J` uses smooth travel and `distance_J` uses ordinary path length with the same weights.

## What the backward pass computes

For the state $y=(x,\ell)$ with $\ell_j=\log(F_j/F_j(0))$, denote the dynamics by $D$ and the running cost by $L$. Using the maximum convention, the Hamiltonian is $H=p^\top D-L$. The forward pass computes positions and fish stocks. The backward pass computes costates $\dot p=-\nabla_y H$: the sensitivities needed to value an admissible movement now in light of the remaining trip.

Final positions are free, so $p_{x_i}(T)=0$. The final fish penalty gives $p_{\ell_j}(T)=-\delta F_j(T)$. These log-stock costates are constant along a planned schedule: $\dot\ell_j=-r\sum_i C_{ij}$, and neither the boat velocity at fixed target and throttle nor the running cost depends on $\ell_j$. The values still affect the position costates through fishing intensity.

The solver differentiates the forward flow, updates the target and throttle using adjoint gradients and Hamiltonian comparisons, and checks the controls against the hull and speed constraints. A projected residual below 0.001 and an integrated Hamiltonian gap below 0.005 on a denser time grid are numerical acceptance criteria. Feedback replays check that bounded attraction weights reproduce the planned velocities and that integrating them reproduces the positions, stocks and distance-based cost. These checks apply to the regularized control problem described here.
