#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

"""
Single-rigid-body (centroidal) quadruped model of the nominal gradient-based NMPC of Quadruped-PyMPC
(IIT DLS lab, github.com/iit-DLSLab/Quadruped-PyMPC, BSD-3-Clause; files
quadruped_pympc/controllers/gradient/nominal/centroidal_{model,nmpc}_nominal.py), re-implemented
here with the same equations, parameter layout, weights and friction cone. At upstream commit 6dcbb35
the dynamics agree with the upstream model file to the last bit on 2000 random and walking states.

States (30): CoM position (3), CoM velocity (3), roll/pitch/yaw, base angular velocity (3), foot
positions FL FR RL RR (12), six "integral" states. Inputs (24): foot velocities (12), ground reaction
forces (12, world frame). Parameters (83): the 29 model parameters of Quadruped-PyMPC (stance flags,
mu, stance proximity, base position, base yaw, external wrench, inertia, mass) followed by the stage
reference y_ref = [x_ref; u_ref] (54), so that every cost variant can be EXTERNAL.

Faithfully kept upstream quirk: the integral-state derivatives are x_int + (z, vx, vy, vz, roll,
pitch), not the tracking error, because the upstream code adds to a copy of the state slice.
"""

import casadi as ca
import numpy as np
from acados_template import AcadosModel

NX, NU = 30, 24
NY = NX + NU
NP_MODEL = 29
NP = NP_MODEL + NY
NH = 20                      # friction cone rows: 5 per foot
GRAVITY = 9.81
DT = 0.02
N_HORIZON = 12

P_STANCE = slice(0, 4)
P_MU = 4
P_PROX = slice(5, 9)
P_BASE_POS = slice(9, 12)
P_BASE_YAW = 12
P_WRENCH = slice(13, 19)
P_INERTIA = slice(19, 28)
P_MASS = 28
P_YREF = slice(NP_MODEL, NP_MODEL + NY)

ACADOS_INFTY = 1e10


def weights(robot='go2'):
    """Q (30) and R (24) of Centroidal_NMPC_Nominal.set_weight."""
    q = np.concatenate(([0, 0, 1500], [200, 200, 200], [500, 500, 0], [20, 20, 50],
                        [300, 300, 300]*4, [50], [10], [10], [10], [10], [10])).astype(float)
    r_force = [1e-5, 1e-5, 1e-5] if robot == 'hyqreal' else [1e-3, 1e-3, 1e-3]
    r = np.concatenate(([1e-4, 1e-4, 1e-5]*4, r_force*4)).astype(float)
    return np.diag(q), np.diag(r)


def f_cont(x, u, p, foothold_optimization=True):
    """Centroidal_Model_Nominal.forward_dynamics."""
    foot_vel = [u[3*i:3*i+3] for i in range(4)]
    foot_force = [u[12+3*i:15+3*i] for i in range(4)]
    com = x[0:3]
    lin_vel = x[3:6]
    roll, pitch, yaw = x[6], x[7], x[8]
    w = x[9:12]
    feet = [x[12+3*i:15+3*i] for i in range(4)]
    stance = [p[i] for i in range(4)]
    prox = [p[5+i] for i in range(4)]
    f_ext = p[13:16]
    tau_ext = p[16:19]
    inertia = ca.reshape(p[P_INERTIA], 3, 3)
    mass = p[P_MASS]

    force = foot_force[0]*stance[0] + foot_force[1]*stance[1] + foot_force[2]*stance[2] + foot_force[3]*stance[3]
    force = force + f_ext
    lin_acc = force/mass + np.array([0.0, 0.0, -GRAVITY])

    conj = ca.SX.eye(3)
    conj[1, 1] = ca.cos(roll)
    conj[2, 2] = ca.cos(pitch)*ca.cos(roll)
    conj[2, 1] = -ca.sin(roll)
    conj[0, 2] = -ca.sin(pitch)
    conj[1, 2] = ca.cos(pitch)*ca.sin(roll)
    euler_rates = ca.inv(conj) @ w

    Rx = ca.SX.eye(3)
    Rx[1, 1] = ca.cos(roll)
    Rx[1, 2] = ca.sin(roll)
    Rx[2, 1] = -ca.sin(roll)
    Rx[2, 2] = ca.cos(roll)
    Ry = ca.SX.eye(3)
    Ry[0, 0] = ca.cos(pitch)
    Ry[0, 2] = -ca.sin(pitch)
    Ry[2, 0] = ca.sin(pitch)
    Ry[2, 2] = ca.cos(pitch)
    Rz = ca.SX.eye(3)
    Rz[0, 0] = ca.cos(yaw)
    Rz[0, 1] = ca.sin(yaw)
    Rz[1, 0] = -ca.sin(yaw)
    Rz[1, 1] = ca.cos(yaw)
    b_R_w = Rx @ Ry @ Rz

    torque = ca.skew(feet[0] - com) @ foot_force[0]*stance[0]
    for i in range(1, 4):
        torque = torque + ca.skew(feet[i] - com) @ foot_force[i]*stance[i]
    torque = torque + tau_ext
    ang_acc = ca.inv(inertia) @ (b_R_w @ torque - ca.skew(w) @ inertia @ w)

    if not foothold_optimization:
        foot_vel = [0.0*v for v in foot_vel]
    foot_pos_dot = [foot_vel[i]*(1 - stance[i])*(1 - prox[i]) for i in range(4)]
    integral_dot = x[24:30] + ca.vertcat(x[2], x[3], x[4], x[5], roll, pitch)
    return ca.vertcat(lin_vel, lin_acc, euler_rates, ang_acc, *foot_pos_dot, integral_dot)


def rk4(x, u, p, dt):
    """One RK4 step, identical to acados' default ERK (4 stages, 1 step) used upstream."""
    k1 = f_cont(x, u, p)
    k2 = f_cont(x + 0.5*dt*k1, u, p)
    k3 = f_cont(x + 0.5*dt*k2, u, p)
    k4 = f_cont(x + dt*k3, u, p)
    return x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def friction_cone(u, mu):
    """Centroidal_NMPC_Nominal.create_friction_cone_constraints: flat-ground pyramid, 5 rows per foot
    [f_x - mu f_z <= 0, f_y - mu f_z <= 0, f_y + mu f_z >= 0, f_x + mu f_z >= 0, f_min <= f_z <= f_max]."""
    rows = []
    for i in range(4):
        f = u[12+3*i:15+3*i]
        rows += [f[0] - mu*f[2], f[1] - mu*f[2], f[1] + mu*f[2], f[0] + mu*f[2], f[2]]
    return ca.vertcat(*rows)


def friction_cone_bounds(f_max, f_min=0.0):
    lh = np.tile([-ACADOS_INFTY, -ACADOS_INFTY, 0.0, 0.0, f_min], 4)
    uh = np.tile([0.0, 0.0, ACADOS_INFTY, ACADOS_INFTY, f_max], 4)
    return lh, uh


def export_model(name, cost_type, dt=DT, robot='go2'):
    """cost_type 'EXTERNAL': dt*0.5*||y - y_ref||_W^2 per stage and 0.5*||x - x_ref||_Q^2 at the end,
    with y_ref from the parameters (use cost_scaling = ones). 'LINEAR_LS': no cost expression, the
    caller sets W, Vx, Vu and y_ref (acados' default cost_scaling = dt gives the same NLP)."""
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)
    p = ca.SX.sym('p', NP)
    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = x, u, p
    model.disc_dyn_expr = rk4(x, u, p, dt)
    h = friction_cone(u, p[P_MU])
    model.con_h_expr = h
    model.con_h_expr_0 = h
    if cost_type == 'EXTERNAL':
        Q, R = weights(robot)
        W = np.block([[Q, np.zeros((NX, NU))], [np.zeros((NU, NX)), R]])
        r = ca.vertcat(x, u) - p[P_YREF]
        re = x - p[NP_MODEL:NP_MODEL+NX]
        model.cost_expr_ext_cost = dt*0.5*(r.T @ W @ r)
        model.cost_expr_ext_cost_e = 0.5*(re.T @ Q @ re)
    return model
