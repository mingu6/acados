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

% FILTERDDP through the MATLAB/Octave interface: the pendulum on cart of getting_started, stabilized from
% near the upright with an active force limit, with a NONLINEAR_LS
% cost, ERK dynamics and a nonlinear inequality on the control, solved with FILTERDDP (exact Hessian and
% Gauss-Newton) and Gauss-Newton SQP as in getting_started, and with the force limit softened. Checks that the
% solutions agree, that warm_start_from_policy warm starts a receding horizon step and refuses without a policy,
% and that an option FILTERDDP does not use is rejected.

import casadi.*
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'getting_started'));
check_acados_requirements()

N = 20;
T = 1;
x0 = [0; 0.5; 0; 0];  % near the upright equilibrium, where the solution is unique

tags = {'filterddp_exact', 'filterddp_gauss_newton', 'sqp_gauss_newton', 'filterddp_soft', 'sqp_soft'};
x = cell(size(tags));
u = cell(size(tags));
solvers = cell(size(tags));
for k = 1:numel(tags)
    ocp = filterddp_pendulum_ocp(N, T, x0, tags{k});
    solver = AcadosOcpSolver(ocp);
    solver.set('constr_x0', x0);
    if strcmp(tags{k}, 'sqp_soft')
        % from the FILTERDDP solution of the soft problem, which SQP must accept as a solution
        solver.set('init_x', x{4});
        solver.set('init_u', u{4});
    else
        solver.set('init_x', zeros(4, N+1));
        solver.set('init_u', zeros(1, N));
    end
    solver.solve();
    status = solver.get('status');
    fprintf('%s: status %d, %d iterations, cost %.10f\n', tags{k}, status, solver.get('sqp_iter'), solver.get_cost());
    assert(status == 0, sprintf('%s failed with status %d', tags{k}, status));
    x{k} = solver.get('x');
    u{k} = solver.get('u');
    solvers{k} = solver;
end
n_active = sum(abs(u{3}(:)) > 20 - 1e-4);
fprintf('%d stages at the force limit\n', n_active);
assert(n_active > 0, 'force limit not active');
for pair = {[1, 3], [2, 3], [4, 5]}
    k = pair{1}(1);
    r = pair{1}(2);
    err_x = max(abs(x{k}(:) - x{r}(:)));
    err_u = max(abs(u{k}(:) - u{r}(:)));
    fprintf('%s against %s: max error x %.2e, u %.2e\n', tags{k}, tags{r}, err_x, err_u);
    assert(err_x < 1e-4 * max(1, max(abs(x{r}(:)))) && err_u < 1e-4 * max(1, max(abs(u{r}(:)))), ...
        sprintf('%s solution differs from %s', tags{k}, tags{r}));
end
assert(max(abs(u{5}(:))) > 20 + 1e-2, 'the soft force limit is not exceeded');

% warm_start_from_policy: receding horizon steps from the predicted next state, warm started from the policy of the
% previous solve (exact Hessian), against cold solves from zero (Gauss-Newton FILTERDDP)
warm = solvers{1};
cold = solvers{2};
for step = 1:5
    x1 = warm.get('x', 1);
    warm.set('constr_x0', x1);
    status = warm.warm_start_from_policy(x1);
    assert(status == 0, sprintf('warm_start_from_policy returned %d at step %d', status, step));
    assert(isequal(warm.get('x', 0), x1), 'warm_start_from_policy did not start the rollout from x0');
    status = warm.warm_start_from_policy(x1);
    assert(status == 5, sprintf('a second warm_start_from_policy returned %d, expected 5 (no policy)', status));
    warm.solve();
    n_warm = warm.get('sqp_iter');
    cold.set('constr_x0', x1);
    cold.set('init_x', zeros(4, N+1));
    cold.set('init_u', zeros(1, N));
    cold.solve();
    assert(warm.get('status') == 0 && cold.get('status') == 0, sprintf('step %d: a solve failed', step));
    u_warm = warm.get('u');
    u_cold = cold.get('u');
    err_u = max(abs(u_warm(:) - u_cold(:)));
    fprintf('warm start step %d: %d iterations (cold %d), max error u %.2e\n', step, n_warm, cold.get('sqp_iter'), err_u);
    assert(err_u < 1e-4 * max(1, max(abs(u_cold(:)))), sprintf('step %d: warm and cold solutions differ', step));
end
warm.reset();
status = warm.warm_start_from_policy(x1);
assert(status == 5, sprintf('warm_start_from_policy after reset returned %d, expected 5 (no policy)', status));

% options of the SQP-type solvers that FILTERDDP does not use are rejected
ocp = filterddp_pendulum_ocp(N, T, x0, 'filterddp_exact');
ocp.solver_options.qp_solver_iter_max = 100;
rejected = false;
try
    ocp.make_consistent();
catch err
    rejected = ~isempty(strfind(err.message, 'qp_solver_iter_max'));
    fprintf('qp_solver_iter_max rejected: %s\n', err.message);
end
assert(rejected, 'qp_solver_iter_max was not rejected');
disp('all checks passed');

