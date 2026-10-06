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

% FILTERDDP on a multi-phase OCP through the MATLAB/Octave interface: the transition example of
% mocp_transition_example, a double integrator (nx = 2) with state bounds, a transition stage without
% controls (nu = 0) and a single integrator (nx = 1) without its terminal bound, solved with FILTERDDP and the
% exact-Hessian SQP solver.
% Checks that the solutions agree. The tolerance is tight because the small acceleration cost determines the
% controls only weakly.

import casadi.*
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'mocp_transition_example'));
check_acados_requirements()

settings = get_example_settings();
N_list = [10, 1, 15];
N_horizon = sum(N_list);
tol = 1e-10;

solvers = {'FILTERDDP', 'SQP'};
x = cell(size(solvers));
u = cell(size(solvers));
for k = 1:numel(solvers)
    ocp = AcadosMultiphaseOcp(N_list);
    ocp.set_phase(formulate_double_integrator_ocp(settings, 1), 1);
    transition = AcadosOcp();
    transition.model = get_transition_model();
    transition.cost.cost_type = 'NONLINEAR_LS';
    transition.model.cost_y_expr = transition.model.x;
    transition.cost.W = diag([settings.L2_COST_P, 1e-1 * settings.L2_COST_V]);
    transition.cost.yref = zeros(2, 1);
    ocp.set_phase(transition, 2);
    phase_3 = formulate_single_integrator_ocp(settings, 1);
    % FILTERDDP does not support terminal constraints
    phase_3.constraints.idxbx_e = [];
    phase_3.constraints.lbx_e = [];
    phase_3.constraints.ubx_e = [];
    ocp.set_phase(phase_3, 3);
    ocp.mocp_opts.integrator_type = {'ERK', 'DISCRETE', 'ERK'};

    T_HORIZON_1 = 0.4 * settings.T_HORIZON;
    T_HORIZON_2 = settings.T_HORIZON - T_HORIZON_1;
    ocp.solver_options.tf = settings.T_HORIZON;
    ocp.solver_options.time_steps = [T_HORIZON_1 / N_list(1) * ones(1, N_list(1)), 0.0, ...
                                    T_HORIZON_2 / N_list(3) * ones(1, N_list(3))];
    ocp.solver_options.cost_scaling = [T_HORIZON_1 / N_list(1) * ones(1, N_list(1)), 1.0, ...
                                      T_HORIZON_2 / N_list(3) * ones(1, N_list(3)), 1.0];
    ocp.solver_options.nlp_solver_type = solvers{k};
    ocp.solver_options.hessian_approx = 'EXACT';
    ocp.solver_options.nlp_solver_max_iter = 300;
    ocp.solver_options.nlp_solver_tol_stat = tol;
    ocp.solver_options.nlp_solver_tol_eq = tol;
    ocp.solver_options.nlp_solver_tol_ineq = tol;
    ocp.solver_options.nlp_solver_tol_comp = tol;
    if strcmp(solvers{k}, 'FILTERDDP')
        ocp.solver_options.regularize_method = 'NO_REGULARIZE';
        ocp.solver_options.globalization = 'FIXED_STEP';
    else
        ocp.solver_options.regularize_method = 'MIRROR';
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING';
    end
    ocp.name = ['mocp_', lower(solvers{k})];
    ocp.code_gen_options.code_export_directory = ['c_generated_code_', ocp.name];
    ocp.code_gen_options.json_file = [ocp.name, '.json'];

    solver = AcadosOcpSolver(ocp);
    solver.solve();
    status = solver.get('status');
    fprintf('%s: status %d, %d iterations, cost %.10f\n', solvers{k}, status, solver.get('sqp_iter'), solver.get_cost());
    assert(status == 0, sprintf('%s failed with status %d', solvers{k}, status));
    x{k} = [];
    u{k} = [];
    for i = 0:N_horizon
        x{k} = [x{k}; solver.get('x', i)];
    end
    for i = 0:N_horizon-1
        u{k} = [u{k}; solver.get('u', i)];
    end
end
err_x = max(abs(x{1} - x{2}));
err_u = max(abs(u{1} - u{2}));
fprintf('FILTERDDP against SQP: max error x %.2e, u %.2e\n', err_x, err_u);
assert(err_x < 1e-6 * max(1, max(abs(x{2}))) && err_u < 1e-6 * max(1, max(abs(u{2}))), ...
    'FILTERDDP solution differs from SQP');
disp('all checks passed');
