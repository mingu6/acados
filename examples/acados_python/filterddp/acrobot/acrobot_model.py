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

import casadi as ca
import numpy as np
from acados_template import AcadosModel

DT = 0.05
NX = 4
NP = 11

# parameters: m1 I1 l1 lc1 m2 I2 l2 lc2 g b1 b2
DEFAULT_PARAMETERS = np.array([1.0, 0.333, 1.0, 0.5, 1.0, 0.333, 1.0, 0.5, 9.81, 0.0, 0.0])


def acrobot_explicit(p, x, u):
    mass1, inertia1, length1, lengthcom1 = p[0], p[1], p[2], p[3]
    mass2, inertia2, length2, lengthcom2 = p[4], p[5], p[6], p[7]
    gravity, friction1, friction2 = p[8], p[9], p[10]

    q = x[0:2]
    v = x[2:4]

    m11 = inertia1 + inertia2 + mass2*length1*length1 + 2.0*mass2*length1*lengthcom2*ca.cos(q[1])
    m12 = inertia2 + mass2*length1*lengthcom2*ca.cos(q[1])
    m22 = inertia2
    det = m11*m22 - m12*m12
    minv11 = m22/det
    minv12 = -m12/det
    minv22 = m11/det

    tau1 = -mass1*gravity*lengthcom1*ca.sin(q[0]) - mass2*gravity*(length1*ca.sin(q[0]) + lengthcom2*ca.sin(q[0] + q[1]))
    tau2 = -mass2*gravity*lengthcom2*ca.sin(q[0] + q[1])

    c11 = -2.0*mass2*length1*lengthcom2*ca.sin(q[1])*v[1]
    c12 = -mass2*length1*lengthcom2*ca.sin(q[1])*v[1]
    c21 = mass2*length1*lengthcom2*ca.sin(q[1])*v[0]

    r1 = -(c11*v[0] + c12*v[1]) + tau1 - friction1*v[0]
    r2 = -(c21*v[0]) + tau2 + u[0] - friction2*v[1]

    qdd1 = minv11*r1 + minv12*r2
    qdd2 = minv12*r1 + minv22*r2
    return ca.vertcat(v[0], v[1], qdd1, qdd2)


def acrobot_rk4(p, x, u):
    k1 = acrobot_explicit(p, x, u)
    k2 = acrobot_explicit(p, x + 0.5*DT*k1, u)
    k3 = acrobot_explicit(p, x + 0.5*DT*k2, u)
    k4 = acrobot_explicit(p, x + DT*k3, u)
    return x + (DT/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def export_acrobot_rk4_model() -> AcadosModel:
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', 1)
    p = ca.SX.sym('p', NP)

    model = AcadosModel()
    model.name = 'acrobot_rk4'
    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = acrobot_rk4(p, x, u)

    x_target = np.array([np.pi, 0.0, 0.0, 0.0])
    Q = np.diag([1.0, 1.0, 0.1, 0.1])
    dx = x - x_target
    model.cost_expr_ext_cost = DT*(2.0*ca.dot(u, u) + dx.T @ Q @ dx)
    model.cost_expr_ext_cost_e = 1000.0*dx.T @ Q @ dx
    return model

