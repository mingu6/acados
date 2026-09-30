%
% Copyright (c) The acados authors.
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;
%

function ocp = filterddp_pendulum_ocp(N, T, x0, tag)
    % pendulum on cart OCP of getting_started for FILTERDDP (tag filterddp_exact, filterddp_gauss_newton,
    % filterddp_soft) or SQP (sqp_gauss_newton, sqp_soft); the _soft tags soften the force limit
    model = get_pendulum_on_cart_model();
    nx = length(model.x);
    nu = length(model.u);

    ocp = AcadosOcp();
    ocp.model = model;
    W_x = diag([1e3, 1e3, 1e-2, 1e-2]);
    W_u = 1e-2;
    ocp.cost.cost_type_0 = 'NONLINEAR_LS';
    ocp.cost.W_0 = W_u;
    ocp.cost.yref_0 = zeros(nu, 1);
    ocp.model.cost_y_expr_0 = model.u;
    ocp.cost.cost_type = 'NONLINEAR_LS';
    ocp.cost.W = blkdiag(W_x, W_u);
    ocp.cost.yref = zeros(nx + nu, 1);
    ocp.model.cost_y_expr = vertcat(model.x, model.u);
    ocp.cost.cost_type_e = 'NONLINEAR_LS';
    ocp.model.cost_y_expr_e = model.x;
    ocp.cost.yref_e = zeros(nx, 1);
    ocp.cost.W_e = W_x;

    U_max = 20;  % active on most of the horizon
    ocp.model.con_h_expr = model.u;
    ocp.model.con_h_expr_0 = model.u;
    ocp.constraints.lh = -U_max;
    ocp.constraints.lh_0 = -U_max;
    ocp.constraints.uh = U_max;
    ocp.constraints.uh_0 = U_max;
    ocp.constraints.x0 = x0;
    if endsWith_custom(tag, '_soft')
        % soft force limit with a penalty weak enough to be exceeded at the optimum
        ocp.constraints.idxsh = 0;
        ocp.constraints.idxsh_0 = 0;
        ocp.cost.zl = 2.0;
        ocp.cost.zu = 2.0;
        ocp.cost.Zl = 1.0;
        ocp.cost.Zu = 1.0;
        ocp.cost.zl_0 = 2.0;
        ocp.cost.zu_0 = 2.0;
        ocp.cost.Zl_0 = 1.0;
        ocp.cost.Zu_0 = 1.0;
    end

    opts = ocp.solver_options;
    opts.N_horizon = N;
    opts.tf = T;
    opts.integrator_type = 'ERK';
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
    opts.qp_solver_cond_N = N;
    opts.nlp_solver_max_iter = 300;
    opts.nlp_solver_tol_stat = 1e-8;
    opts.nlp_solver_tol_eq = 1e-8;
    opts.nlp_solver_tol_ineq = 1e-8;
    opts.nlp_solver_tol_comp = 1e-8;
    if strncmp(tag, 'filterddp', 9)
        opts.nlp_solver_type = 'FILTERDDP';
        opts.regularize_method = 'NO_REGULARIZE';
        opts.globalization = 'FIXED_STEP';
        if strcmp(tag, 'filterddp_soft')
            opts.hessian_approx = 'EXACT';
        elseif strcmp(tag, 'filterddp_gauss_newton')
            opts.hessian_approx = 'GAUSS_NEWTON';
            % FILTERDDP options through code generation
            opts.filterddp_mu_init = 0.1;
            opts.filterddp_symmetric_value_hessian = false;
        else
            opts.hessian_approx = 'EXACT';
        end
    else
        % settings of getting_started/minimal_example_ocp.m
        opts.nlp_solver_type = 'SQP';
        opts.hessian_approx = 'GAUSS_NEWTON';
        opts.globalization = 'MERIT_BACKTRACKING';
        opts.qp_solver_mu0 = 1e3;
        opts.qp_solver_cond_N = 5;
    end
    ocp.model.name = ['pendulum_', tag];
    ocp.name = ocp.model.name;
    ocp.code_gen_options.code_export_directory = ['c_generated_code_', tag];
    ocp.code_gen_options.json_file = [tag, '_ocp.json'];
end
