# Contact-Implicit Spring Flamingo Walking OCP

This document defines and reproduces a planar Spring Flamingo walking gait
in the spirit of Posa, Cantu, and Tedrake (2013, "A Direct Method for
Trajectory Optimization of Rigid Bodies Through Contact", Secs. 3 and 4.3).
The problem minimizes the mechanical cost of transport of one periodic half
stride. It uses no mode schedule and no reference trajectory:

- the contact sequence emerges from complementarity constraints;
- the swing foot is kept off the ground by a speed-dependent clearance rule.

The mechanics follow the variational contact-implicit form of the
FilterDDPpinocchio project (`docs/contact_implicit_biped_ocp.md`). Pinocchio
evaluates kinematics and inverse dynamics through its CasADi scalar type, and
the acados FilterDDP solver solves the OCP. The implementation is in this
directory.

## Sources and revisions

| Source | Role | Revision |
| --- | --- | --- |
| Posa, Cantu, Tedrake 2013 | Problem, complementarity contact, cost of transport, mirror periodicity, Figs. 5-6 | Sept 4, 2013 preprint |
| `~/stuff/ContactImplicitMPC.jl` | Flamingo parameters and kinematics; gait `gait_forward_36_4.jld2` for the start state, stride, and model-parity tests | `989c8e6` |
| FilterDDPpinocchio | Variational mechanics and maximum-dissipation friction (`docs/contact_implicit_biped_ocp.md`) | `8572095` + working tree |
| This repository (branch `filterddp`) | FilterDDP NLP solver in acados | `da2bd8bd0` on upstream `00947ddc3` |
| Pinocchio | Rigid-body model, RNEA, frames, CasADi scalar type | conda-forge `pinocchio-python` 4.1.0 |
| CasADi | Expressions and code generation | 3.7.2 |

Posa gives no numerical Spring Flamingo parameters, so the robot is
ContactImplicitMPC.jl's `flamingo` model. Posa's cost-of-transport values
(0.18 nominal, 0.04 optimized) are for a different parameter set and are not
a target.

## Robot model

### Physical parameters

All seven bodies are planar rigid links in the $x$-$z$ plane. Gravity is
$g=9.81$ m/s$^2$ along $-z$. The values are from
`ContactImplicitMPC.jl/src/dynamics/flamingo/model.jl:452-495`:

| Body | Mass [kg] | Pitch inertia about COM [kg m$^2$] | Length [m] | COM offset from proximal joint [m] |
| --- | ---: | ---: | ---: | ---: |
| Torso | 12.0 | 0.10 | 0.385 (drawing only) | 0.20 (up from hip) |
| Thigh (×2) | 0.4598 | 0.01256 | 0.42 (hip→knee) | 0.21 |
| Calf (×2) | 0.306 | 0.00952 | 0.45 (knee→ankle) | 0.225 |
| Foot (×2) | 0.3466 | 0.0015 | 0.1725 (ankle→toe), 0.0525 (ankle→heel) | 0.06 toward toe |

The total mass is $m=14.2248$ kg and the weight $mg=139.55$ N. The model has
no springs, joint damping, or joint friction; Posa also omits the passive
elasticity. The URDF shipped with ContactImplicitMPC.jl does not match these
parameters and is not used.

### Pinocchio construction and coordinates

The model is built programmatically (`flamingo_model.build_model`). Joint
order fixes the configuration

$$
q=(x,z,\theta,\phi_{h1},\phi_{k1},\phi_{a1},\phi_{h2},\phi_{k2},\phi_{a2})\in\mathbb R^9,
$$

where:

- $(x,z)$ is the hip point, from prismatic joints;
- $\theta$ is the torso pitch;
- $\phi_{h},\phi_{k},\phi_{a}$ are the hip, knee, and ankle angles of legs 1
  and 2.

| Joint | Type | Parent | Placement in parent | Body COM in joint frame |
| --- | --- | --- | --- | --- |
| `root_x`, `root_z` | prismatic $x$, $z$ | universe, `root_x` | identity | none |
| `pitch` | revolute, axis $-y$ | `root_z` | identity | torso $(0,0,+0.20)$ |
| `hip1`, `hip2` | revolute, axis $-y$ | `pitch` | identity | thigh $(0,0,-0.21)$ |
| `knee1`, `knee2` | revolute, axis $-y$ | hip | $(0,0,-0.42)$ | calf $(0,0,-0.225)$ |
| `ankle1`, `ankle2` | revolute, axis $-y$ | knee | $(0,0,-0.45)$ | foot $(0.06,0,0)$ |

The $-y$ axes make a positive angle rotate $+x$ toward $+z$ (the Julia sign
convention). Each body has inertia `Inertia(m, com, J I_3)`; only the $yy$
entry enters planar dynamics. Every joint is prismatic or revolute, so
$n_q=n_v=9$ and the configuration space is Euclidean. The contact points are
operational frames on the ankles:

| Frame | Parent joint | Local placement |
| --- | --- | --- |
| `toe1`, `toe2` | `ankle1`, `ankle2` | $(+0.1725,0,0)$ |
| `heel1`, `heel2` | `ankle1`, `ankle2` | $(-0.0525,0,0)$ |

The sole lies along the foot frame's $x$ axis, so a zero ankle angle holds
the foot perpendicular to the calf. $p_i(q)=(p^x_i,p^z_i)$ denotes the
position of contact $i\in\{\text{toe1},\text{heel1},\text{toe2},\text{heel2}\}$,
and $J_i(q)=\partial p_i/\partial q\in\mathbb R^{2\times9}$ its Jacobian.
$J_i$ equals the $x,z$ rows of Pinocchio's `LOCAL_WORLD_ALIGNED` frame
Jacobian because $q$ is Euclidean.

Mirror periodicity swaps the legs:

$$
Pq=(x,z,\theta,\phi_{h2},\phi_{k2},\phi_{a2},\phi_{h1},\phi_{k1},\phi_{a1}).
$$

The Julia model uses absolute link angles $q^{\mathrm{abs}}$. The two
conventions are related by the constant affine map $q=Tq^{\mathrm{abs}}+c$,
where $T$ has rows $x$, $z$, $\theta_t$, $\theta_{th1}-\theta_t$,
$\theta_{c1}-\theta_{th1}$, $\theta_{f1}-\theta_{c1}$ and the same for leg 2,
and $c$ carries $-\pi/2$ on both ankles. This map is used only to import the
gait and in the model-parity tests.

## Horizon, state, and dynamics

The time step $h=0.0156728$ s and the stride come from the gait. The stored
gait has 70 intervals and advances $0.23172$ m, as two mirror-symmetric
steps. The OCP covers one step:

$$
N=35,\qquad Nh=0.5485\ \mathrm{s},\qquad
d=\tfrac12\cdot0.23172=0.11586\ \mathrm{m},\qquad \bar v=0.211\ \mathrm{m/s}.
$$

The state holds two consecutive configurations, and the next configuration
is a control. This gives explicit shift dynamics, which is the form acados
FilterDDP requires:

$$
x_k=(q_{k-1},q_k)\in\mathbb R^{18},\qquad
x_{k+1}=f(x_k,u_k)=(q_k,\;q_{k+1}).
$$

The start state $x_0=(q_0,q_1)$ is fixed to the first two gait knots.
acados FilterDDP requires a fixed $x_0$.

## Controls

Each stage has $n_u=52$ controls. Contacts are ordered toe1, heel1, toe2,
heel2.

| Indices | Variable | Dim. | Bounds |
| --- | --- | ---: | --- |
| 0-5 | Joint torques $\tau$ (hip1, knee1, ankle1, hip2, knee2, ankle2) | 6 | $[-100,100]$ N m |
| 6-14 | Next configuration $q_{k+1}$ | 9 | $x,z$ free; $\theta\in[-\frac\pi2,\frac\pi2]$; hips $[-1.25,1.25]$; knees $[-\frac\pi2,\frac\pi2]$; ankles $[-1.1,1.1]$ |
| 15-22 | Contact impulses $p_i=(p_{t,i},p_{n,i})$ | 8 | $p_{n,i}\ge0$ |
| 23-26 | Normal complementarity relaxations $\sigma_{n,i}$ | 4 | $\ge0$ |
| 27-34 | Friction KKT duals $\lambda_i^\pm$ | 8 | $\ge0$ |
| 35-42 | Friction complementarity relaxations $\sigma_{f,i}^\pm$ | 8 | $\ge0$ |
| 43-51 | Periodicity dummy controls $\zeta$ | 9 | free |

That makes 37 bounded controls. Impulses are forces integrated over one
interval.

## Constraints

All rows are functions of $(x_k,u_k)$ and a per-stage parameter
$(q^\star_k,f_k)$, the periodic target and its flag. With

$$
\bar q_k=\tfrac12(q_k+q_{k+1}),\qquad
v_k=\frac{q_{k+1}-q_k}{h},\qquad
a_k=\frac{q_{k-1}-2q_k+q_{k+1}}{h^2},
$$

each stage imposes the following.

**Mechanics** (9 equalities): midpoint inverse dynamics integrated over the
interval,

$$
h\left[\mathrm{RNEA}(\bar q_k,v_k,a_k)-S^T\tau_k\right]-\sum_{i=1}^4J_i(\bar q_k)^Tp_i=0,
$$

where $S\in\{0,1\}^{6\times9}$ selects the six actuated joints. RNEA
includes gravity. This is the scheme of FilterDDPpinocchio's
`docs/contact_implicit_biped_ocp.md`.

**Unilateral contact** (4 equalities, 4 inequalities): the gap is the height
of each contact at the next knot on flat ground,

$$
g_i=p_i^z(q_{k+1})\ge0,\qquad p_{n,i}\,g_i-\sigma_{n,i}=0,\qquad p_{n,i},\sigma_{n,i}\ge0.
$$

**Maximum-dissipation friction** (12 equalities, 8 inequalities): Coulomb
friction solves $\min_{p_t}v_tp_t$ subject to $|p_t|\le\mu p_n$. With
tangential velocity $v_{t,i}=[p_i^x(q_{k+1})-p_i^x(q_k)]/h$ and $\mu=0.74$,
its relaxed KKT conditions are

$$
s_i^\pm=\mu p_{n,i}\mp p_{t,i}\ge0,\qquad
v_{t,i}+\lambda_i^+-\lambda_i^-=0,\qquad
\lambda_i^\pm s_i^\pm-\sigma^\pm_{f,i}=0.
$$

A sliding contact is therefore pushed to the cone face opposing its motion,
while a sticking contact may carry any impulse inside the cone.

**Speed-dependent clearance** (4 inequalities): a contact point moving
horizontally must be off the ground in proportion to its speed,

$$
g_i-\tau_c\left(\sqrt{v_{t,i}^2+\delta^2}-\delta\right)\ge0,\qquad
\tau_c=0.02\ \mathrm{s},\quad \delta=0.01\ \mathrm{m/s}.
$$

A stance point ($v_{t,i}=0$) is left with $g_i\ge0$, so the rule needs no
contact schedule and does not conflict with complementarity. A swinging
point at speed $v$ must be at least $\approx\tau_c|v|$ high; at this gait's
peak swing speed of about 0.9 m/s that is about 2 cm. Touchdown must be
nearly vertical, and slip under load is suppressed. Without this row the
cost-of-transport optimum skims the swing foot along the ground. Posa
reports the same effect for Spring Flamingo in the Fig. 5 caption.

**Mirror periodicity** (9 equalities): the end state must be the mirrored
start state advanced by one step, $x_N=(Pq_0+de_x,\;Pq_1+de_x)$. acados
FilterDDP supports no terminal constraints, so this is imposed on the
controls $q_{N-1}$ and $q_N$ of stages $N-2$ and $N-1$. Every stage carries
the same nine rows:

$$
f_k\left(q_{k+1}-q^\star_k\right)+(1-f_k)\,\zeta_k=0,\qquad
f_k=\begin{cases}1,&k\in\{N-2,N-1\}\\0,&\text{otherwise.}\end{cases}
$$

Where $f_k=1$ the row pins $q_{k+1}=q^\star_k$; elsewhere it pins
$\zeta_k=0$. Either way the block is an identity in distinct control
columns. The targets are

$$
q^\star_{N-2}=Pq_0+de_x+\ell_0e_z,\qquad
q^\star_{N-1}=Pq_1+de_x+\ell_1e_z,\qquad
\ell_j=\max\!\Big(0,\;g_{\min}-\min_i p_i^z(Pq_j+de_x)\Big).
$$

The lift $\ell_j$ with $g_{\min}=2\times10^{-6}$ m keeps every gap at a
pinned knot strictly positive. A pinned knot cannot move, so a zero gap
would leave its gap inequality with no interior. The stored gait has gaps
of 0 and $-1.5\times10^{-9}$ there, so the lift is about 2 µm.

**Summary and rank.** Each stage has 34 equalities and 16 inequalities in
`con_h_expr`, in that row order. acados FilterDDP eliminates the equalities
per stage with a null-space method. This needs $n_{\mathrm{eq}}\le n_u$
($34\le52$) and a full-row-rank control Jacobian, which the solver does not
check. The rank is structural:

- the mechanics block contains $\partial(h\,\mathrm{RNEA})/\partial q_{k+1}\approx M(\bar q_k)/h$, which is nonsingular;
- the 16 complementarity and stationarity rows each carry a $-1$ in a
  distinct relaxation or dual column;
- the periodicity rows are identities in $q_{k+1}$ or $\zeta_k$.

`test_equality_control_jacobian_has_full_row_rank` verifies rank 34 for both
flag values; the smallest singular value is about 0.14. Inequalities
receive solver-managed slacks.

## Objective

The running cost is a smooth mechanical cost of transport, Posa's Eq. 37,
plus regularization and relaxation penalties:

$$
\ell_k=\underbrace{\frac{h}{mgd}\sum_{j=1}^{6}\left(\sqrt{(\tau_{k,j}\,\omega_{k,j})^2+\varepsilon^2}-\varepsilon\right)}_{\text{cost of transport}}
+10^{-4}h\lVert\tau_k\rVert^2+10^{-3}\lVert p_k\rVert^2
+10^{6}\left(\lVert\sigma_{n,k}\rVert^2+\lVert\sigma_{f,k}\rVert^2\right)
+2\,\mathbf 1^T\!\begin{bmatrix}\sigma_{n,k}\\\sigma_{f,k}\end{bmatrix}
+\frac{w_sh}{2}\sum_{i=1}^4r_{i,k}^2+10^{-3}\lVert\zeta_k\rVert^2,
$$

where:

- $\omega_k=S\,v_k$ are the actuated joint rates, so $\tau\omega$ is joint power;
- $\varepsilon=0.1$ W smooths $|\tau\omega|$;
- the slip residual is $r_{i,k}=(p_{n,i}/p_{\mathrm{ref}})\,v_{t,i}$ with
  $p_{\mathrm{ref}}=\tfrac14mgh$ and $w_s=1000$, so it weights tangential
  speed by load.

As $\varepsilon\to0$ the first term tends to Posa's
$\frac1{mgd}\sum_k h\sum_j|\omega_{k,j}\tau_{k,j}|$. The reported cost of
transport is always this non-smooth value, recomputed from the solution.

The terminal cost is zero, because periodicity fixes $x_N$. No term refers to
a reference trajectory.

## Initial guess

The guess uses no gait data beyond the boundary state $q_1$ and the periodic
end configuration $q_e=Pq_1+de_x$ (`flamingo_ocp.initial_knots`). For knot
$j$ with $t=j/N$:

- The hip position and pitch are linear: $(x,z,\theta)_j=(1-t)(x,z,\theta)_{q_1}+t\,(x,z,\theta)_{q_e}$.
- The swing foot is the foot whose ankle moves between $q_1$ and $q_e$
  (foot 1 moves 0.232 m; foot 2 stays). Its ankle follows

  $$
  a(t)=a_s+s(\varphi)(a_e-a_s)+16A\varphi^2(1-\varphi)^2e_z,\qquad
  \varphi=\mathrm{clip}\!\left(\tfrac{t-0.15}{0.7},0,1\right),\quad s(\varphi)=3\varphi^2-2\varphi^3 .
  $$

  Here $s$ is the cubic Bezier with control points $(0,0,1,1)$, and the
  bump is a quartic Bernstein polynomial with apex $A=4$ cm. The swing
  occupies the middle 70% of the horizon.
- Both feet keep their absolute sole angle (flat on the ground). Joint
  angles come from planar two-link inverse kinematics on the start pose's
  knee branch.
- Torques and tangential impulses are zero, and normal impulses are
  $10^{-3}$. This follows Posa's zero initial forces, kept strictly
  positive. The other controls are set consistently with the knots:

  - $\sigma_{n,i}=\max(10^{-3},p_{n,i}g_i^+)$;
  - $\lambda_i^\pm=\max(\mp v_{t,i},0)+10^{-3}$;
  - $\sigma^\pm_{f,i}=\max(10^{-3},\lambda^\pm_i\mu p_{n,i})$;
  - $\zeta=0$.

The guess is strongly infeasible: the mechanics residual is 2.27, close to
$mgh=2.18$ because the zero forces leave gravity unbalanced, and the
periodicity residual is $7.7\times10^{-3}$ because the targets carry the
lift. FilterDDP's push-to-interior moves every bounded control at least
$\kappa_1\max(1,|b|)=0.01$ inside its bounds before the first iteration.

## Solver

acados FilterDDP is an interior-point differential dynamic programming
method with a filter line search. It classifies each `con_h_expr` row as an
equality if its lower and upper bounds are equal, and otherwise as an
inequality with a slack. Log barriers act on the control bounds and slacks.
Settings (`flamingo_ocp.make_ocp`):

| Setting | Value |
| --- | --- |
| `nlp_solver_type`, `integrator_type` | `FILTERDDP`, `DISCRETE` |
| `hessian_approx` | `EXACT`: cost Hessian plus multiplier-contracted dynamics and constraint Hessians from CasADi |
| `regularize_method`, `globalization` | `NO_REGULARIZE`, `FIXED_STEP` (FilterDDP has its own inertia correction and line search) |
| `qp_solver` | `PARTIAL_CONDENSING_HPIPM`, `qp_solver_cond_N = N` (required, not called) |
| `nlp_solver_tol_stat` | $10^{-4}$ (the only tolerance FilterDDP reads) |
| `nlp_solver_max_iter` | 1500 |
| `cost_scaling` | $1$ at every node (the cost already carries $h$) |
| `filterddp_mu_init`, `filterddp_reg_1` | 0.1, $10^{-2}$ (set through ctypes; the Python template has no fields) |
| other `filterddp_*` options | C defaults, including `kappa_1 = kappa_2 = 0.01` |

The fork's `pi` export writes the same vector to every stage, so acados
residuals are not used. Every reported check is recomputed in numpy from the
returned trajectory (`contact_implicit_numpy`, `solve_flamingo.evaluate_trajectory`).

## Validation

`python -m pytest tests` runs 25 deterministic tests with fixed seeds:

- `test_flamingo_model.py` (9): the Pinocchio model agrees with a numpy
  transcription of `model.jl`. Contact positions, centre of mass, mass
  matrix ($T^{-T}M^{\mathrm{abs}}T^{-1}$ against CRBA), and Lagrangian agree
  to $10^{-12}$ over 64 random configurations. The contact Jacobians agree
  analytically and with central differences. The Julia actuation matrix
  equals $S^T$ under the torque permutation. The sign conventions and the
  coordinate and mirror maps are checked.
- `test_gait_reconciliation.py` (5): the stored gait satisfies the
  ContactImplicitMPC.jl integrator through Pinocchio to $8.8\times10^{-7}$,
  which establishes model parity. That integrator uses
  $M(\bar q_{k-1})v_{k-1}$ and trapezoidal bias terms, with $J$ at $q_{k+1}$.
  The OCP's RNEA-midpoint residual of the same gait is up to 0.127, or 5.8%
  of $mgh$ at an impact interval. It is verified term by term as the exact
  mass, bias, and Jacobian-point difference of the two schemes, so the gait
  is close to, but not on, the OCP's feasible set. The test also checks the
  gait's mirror symmetry, its contact complementarity at its $\mu=0.1$, and
  its load.
- `test_casadi_constraints.py` (7): every CasADi row matches the numpy
  oracle to $10^{-10}$, and the Jacobians match central differences to
  $10^{-6}$. It also checks the clearance row (inactive at rest, exact
  requirement at 1 m/s), the periodicity rows for both flags, rank 34 of the
  equality control Jacobian, and a symmetric contracted Hessian that is
  exactly zero for zero multipliers.
- `test_solve_smoke.py` (4): the periodic targets, the Bezier guess
  geometry, and interiority and consistency of the initial controls. It then
  runs a full solve and checks convergence, feasibility, periodicity, the
  clearance lower bound, and the toe-push-off mode structure.

## Results

Default solve, on a 12th Gen Intel Core i7-12700 with one thread. The
solve time is the median of three timed solves after a warmup, excluding
code generation.

| Quantity | Value |
| --- | --- |
| Status, iterations, solve time | converged, 282, 1.03 s |
| Barrier path | $\mu$ = 0.1, 0.02, $8\times10^{-4}$, $2.8\times10^{-5}$; peak regularization 98, zero at the end |
| Objective, cost of transport | 0.21348, 0.0810 |
| Max equality violation (numpy) | $2.3\times10^{-8}$ |
| Min inequality, periodicity residual | $2.0\times10^{-6}$ (the pinned gap), $9.6\times10^{-21}$ |
| Max $\sigma_n$, $\sigma_f$ | $9.1\times10^{-6}$, $1.2\times10^{-5}$ |
| Max load-weighted slip speed | 0.0023 m/s |
| Torque range | $[-28.5, 8.4]$ N m (the $\pm100$ N m bound is inactive) |
| Swing apex, peak swing speed | 5.2 cm, 0.91 m/s |

**Contact modes compared with Posa Fig. 6.** `gait_metrics.py` labels each
interval and foot as S (toe and heel loaded), T (toe only), H (heel only), or
W (swing). A point counts as loaded above 1% of body weight. The half stride
is unrolled to a full cycle with its mirror symmetry. Posa's optimized
sequence, read from Fig. 6, is stance 0.50, toe-only 0.16 (push-off 0.13 plus
toe-first touchdown 0.03), heel-only 0, swing 0.34, and double support 0.32,
in the order S-T-W-T over 1.88 s.

| | Stance | Toe only | Heel only | Swing | Double support | Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Posa Fig. 6 | 0.50 | 0.16 | 0.00 | 0.34 | 0.32 | 0 |
| This OCP | 0.50 | 0.17 | 0.01 | 0.31 | 0.37 | 0.10 |

Each foot pushes off on its toe (about 0.1 s), swings for about 0.31 s, and
lands heel-first for one interval before going flat. The distance is the L1
distance of the fractions plus the double-support difference. The cycle is
1.10 s against Posa's 1.88 s, because $h$ and $N$ are the gait's. Each half
starts with a short toe-only phase and a one-interval lift of the leading
foot. This is imposed by $x_0$, which is taken from the gait at a touchdown
instant.

**Local solutions.** The problem is non-convex, and the barrier path
selects one of several nearby local optima. Solves are deterministic for
identical inputs. Perturbing the pinned gap $g_{\min}$ by a few µm, which
changes the targets but not the problem's character, gives:

| $g_{\min}$ [µm] | Iter. | CoT | Swing apex [cm] | Stance / toe / heel / swing | Double support | Distance to Posa |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1.5 | 423 | 0.0675 | 2.7 | 0.49 / 0.19 / 0.01 / 0.31 | 0.37 | 0.13 |
| **2 (default)** | **282** | **0.0810** | **5.2** | **0.50 / 0.17 / 0.01 / 0.31** | **0.37** | **0.10** |
| 2.5 | 419 | 0.0678 | 2.8 | 0.50 / 0.17 / 0.01 / 0.31 | 0.37 | 0.10 |
| 3 | 506 | 0.0684 | 2.5 | 0.54 / 0.14 / 0.00 / 0.31 | 0.37 | 0.14 |
| 4 | 549 | 0.0610 | 2.6 | 0.46 / 0.21 / 0.01 / 0.31 | 0.37 | 0.19 |

All five converge. The contact structure is stable: toe push-off, a swing
fraction of 0.31, and double support of 0.37. The cost of transport
(0.061-0.081) and swing apex (2.5-5.2 cm) vary between local solutions, so
single-run differences of that size are not meaningful. The default run
reaches a higher-stepping local solution than the others.

## Reproduction

```bash
cd examples/acados_python/filterddp/spring_flamingo
conda env create -f environment.yml          # Python 3.12, pinocchio-python 4.1.0, casadi 3.7.2
conda activate flamingo-acados
pip install --no-deps -e ../../../../interfaces/acados_template
export ACADOS_SOURCE_DIR=$(cd ../../../.. && pwd)
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
python -m pytest tests -q                     # 25 tests, including one full solve
./reproduce.sh                                # solution, figure, and spread study
```

`CIMPC_DIR` selects the ContactImplicitMPC.jl checkout; the default is
`~/stuff/ContactImplicitMPC.jl`. `reproduce.sh` always uses this checkout
(override with `FILTERDDP_ACADOS_DIR`): an `ACADOS_SOURCE_DIR` pointing at
upstream acados, which has no FilterDDP, fails at compile time. It writes to `output/`, which is ignored by
git:

- `output/flamingo/`: `trajectory.csv` and `initial_trajectory.csv`
  (configurations, contact positions, torques, and per-contact impulses,
  gaps, clearance margins, relaxations, and duals); `statistics.csv`
  (per-iteration FilterDDP statistics); `summary.json`; and
  `flamingo.png` / `flamingo.gif` (the Posa-style figure and animation).
- `output/spread/gap*/`: the perturbed solves.

The figure shows three panels with the initial guess overlaid as "Initial
Sequence":

- a filmstrip of the full stride with the fan-shaped torso of Posa Fig. 5(a);
- CM height against normalized time, as in Fig. 5(b);
- the per-foot mode sequence, as in Fig. 6.

## Differences from Posa and ContactImplicitMPC.jl

- Robot parameters are ContactImplicitMPC.jl's, not the original Spring
  Flamingo's.
- The mechanics are this repository's midpoint RNEA scheme. Posa uses
  backward Euler on $(q,\dot q)$, and ContactImplicitMPC.jl uses the
  momentum form with $J(q_{k+1})$.
- Friction is the maximum-dissipation KKT system with relaxations penalized
  toward zero. Posa uses the Stewart-Trinkle complementarity with a
  $G_iH_i\le\varepsilon$ continuation.
- The cost of transport is smoothed with $\varepsilon=0.1$ W.
- Swing clearance is imposed by the speed-dependent rule; Posa leaves swing
  height unconstrained.
- The start state is fixed to the gait's and periodicity is imposed on
  stage controls. Posa's initial state is free, with periodicity as a
  boundary constraint.
- The stride is fixed to the gait's, where Posa uses a minimum-stride
  inequality.
- The torque bound of 100 N m and the joint limits are placeholders and are
  inactive at the solution.
