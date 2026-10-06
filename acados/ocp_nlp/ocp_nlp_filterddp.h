/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


/// \addtogroup ocp_nlp
/// @{
/// \addtogroup ocp_nlp_solver
/// @{
/// \addtogroup ocp_nlp_filterddp ocp_nlp_filterddp
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_FILTERDDP_H_
#define ACADOS_OCP_NLP_OCP_NLP_FILTERDDP_H_

#ifdef __cplusplus
extern "C" {
#endif

// acados
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/utils/types.h"



/************************************************
 * options
 ************************************************/

typedef struct
{
    ocp_nlp_opts *nlp_opts;

    double mu_init;             // barrier parameter initialization
    double ineq_dual_init;      // initial value of bound multipliers
    double kappa_1;             // fraction-to-boundary parameters for interior initialization
    double kappa_2;

    double reg_1;               // inertia correction: initial regularization
    double reg_min;
    double reg_max;
    double kappa_bar_w_p;       // inertia correction: increase factor at first regularization
    double kappa_w_p;           // inertia correction: increase factor
    double kappa_w_m;           // inertia correction: decrease factor

    double kappa_eps;           // barrier subproblem tolerance factor
    double kappa_mu;            // linear barrier decrease factor
    double theta_mu;            // superlinear barrier decrease exponent
    double tau_min;             // lower bound on fraction-to-boundary parameter

    double s_max;               // scaling threshold for NLP error
    double eta_L;               // Armijo relaxation factor
    double s_L;                 // switching condition exponent of the barrier model
    double delta;               // switching condition multiplier of constraint violation
    double s_theta;             // switching condition exponent of constraint violation
    double gamma_theta;         // filter margin for constraint violation
    double gamma_L;             // filter margin for barrier function
    double theta_max_factor;    // maximum constraint violation accepted by the filter, relative to initial
    double theta_min_factor;    // constraint violation threshold for the switching condition, relative to initial

    int warm_start;             // 1: initialize each solve from the shifted affine policy and final barrier parameter of the previous solve
    int policy_at_cap;          // a solve stopped at max_iter leaves for the warm start 1 (default): the update rules
                                // of an extra backward pass at the returned iterate; 0: those of the last backward
                                // pass, shifted from the returned iterate with the feedforward not yet taken, 1 - step
    int symmetric_value_hessian; // value function Hessian recursion: 0: C + beta' B + omega' cx as computed,
                                 // 1: the same averaged with its transpose, 2 (default): the factored Schur complement
                                 // form C + (betaY' B + B' betaY) + betaY' H betaY - W' W, symmetric by construction
    int dynamics_multiplier;    // multiplier of the dynamics Hessian contraction: 0: lambda if the problem has
                                // constraint rows, else Vx; 1: lambda; 2: Vx; 3: the one of Vx, lambda with the
                                // smaller inf-norm, per stage; 4: elementwise the entry of smaller magnitude

    double timeout_max_time;    // maximum time the solve may require before timeout is triggered. No timeout if 0.
    ocp_nlp_timeout_heuristic_t timeout_heuristic; // type of heuristic used to predict the time of the next iteration

} ocp_nlp_filterddp_opts;

//
acados_size_t ocp_nlp_filterddp_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_filterddp_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_filterddp_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_filterddp_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_filterddp_opts_set(void *config_, void *opts_, const char *field, void* value);
//
void ocp_nlp_filterddp_opts_set_at_stage(void *config_, void *opts_, size_t stage, const char *field, void* value);



/************************************************
 * memory
 ************************************************/

typedef struct
{
    // nlp memory
    ocp_nlp_memory *nlp_mem;

    // statistics
    double *stat;
    int stat_m;
    int stat_n;

    // constraint classification per stage, fixed at precompute
    int *nul;       // control lower bounds
    int *nuu;       // control upper bounds
    int *nh;        // equality rows (bound rows with lower == upper)
    int *ng;        // inequality rows with slack (box constraints on x and two sided rows)
    int *ngl;       // slack lower bounds
    int *ngu;       // slack upper bounds
    int **idxh;     // index in [bx; g; h] of each equality row
    int **idxg;     // index in [bx; g; h] of each inequality row
    int **idxs_row; // slack index of each row in [bu; bx; g; h], -1 if hard
    int **idxs_g;   // slack index of each inequality row, -1 if hard
    // classification of the solve whose update rules the warm start shifts
    int *nh_prev;
    int *ng_prev;
    int **idxh_prev;
    int **idxg_prev;
    int **idxs_g_prev;
    int **sides_prev; // per inequality row: 1 lower bound, 2 upper bound, 4 soft lower side, 8 soft upper side

    // bounds gathered per stage in solver order
    struct blasfeo_dvec *ul;    // control lower bounds
    struct blasfeo_dvec *uu;    // control upper bounds
    struct blasfeo_dvec *maskul;
    struct blasfeo_dvec *maskuu;
    struct blasfeo_dvec *gl;    // slack lower bounds
    struct blasfeo_dvec *gu;    // slack upper bounds
    struct blasfeo_dvec *maskgl;
    struct blasfeo_dvec *maskgu;
    struct blasfeo_dvec *lsl;    // lower bounds of the lower slacks of soft inequality rows
    struct blasfeo_dvec *lsu;    // lower bounds of the upper slacks
    struct blasfeo_dvec *masksl; // 1 if the lower side of an inequality row is soft
    struct blasfeo_dvec *masksu;

    // iterate: nlp_out holds x, u; dual variables and slacks are solver specific
    struct blasfeo_dvec *s;     // inequality slacks
    struct blasfeo_dvec *phi;   // equality multipliers
    struct blasfeo_dvec *nu;    // inequality multipliers
    struct blasfeo_dvec *zl;    // control lower bound multipliers
    struct blasfeo_dvec *zu;    // control upper bound multipliers
    struct blasfeo_dvec *zsl;   // slack lower bound multipliers
    struct blasfeo_dvec *zsu;   // slack upper bound multipliers
    struct blasfeo_dvec *xil;   // multipliers of the lower bounds of the soft constraint slacks, which are in nlp_out->ux
    struct blasfeo_dvec *xiu;
    // trial iterate
    struct blasfeo_dvec *s_trial;
    struct blasfeo_dvec *phi_trial;
    struct blasfeo_dvec *nu_trial;
    struct blasfeo_dvec *zl_trial;
    struct blasfeo_dvec *zu_trial;
    struct blasfeo_dvec *zsl_trial;
    struct blasfeo_dvec *zsu_trial;
    struct blasfeo_dvec *xil_trial;
    struct blasfeo_dvec *xiu_trial;

    // costate of the last backward pass, stages 0:N
    struct blasfeo_dvec *costate;

    // update rules, feedforward in column 0, feedback in columns 1:nx
    struct blasfeo_dmat *alpha_beta;      // nu x (nx+1)
    struct blasfeo_dmat *alphas_betas;    // ng x (nx+1)
    struct blasfeo_dmat *psih_omegah;     // nh x (nx+1)
    struct blasfeo_dmat *psig_omegag;     // ng x (nx+1)
    struct blasfeo_dmat *chil_zetal;      // nu x (nx+1)
    struct blasfeo_dmat *chiu_zetau;      // nu x (nx+1)
    struct blasfeo_dmat *chisl_zetasl;    // ng x (nx+1)
    struct blasfeo_dmat *chisu_zetasu;    // ng x (nx+1)
    struct blasfeo_dmat *sigl_rule;       // ng x (nx+1), soft constraint slacks
    struct blasfeo_dmat *sigu_rule;
    struct blasfeo_dmat *xil_rule;
    struct blasfeo_dmat *xiu_rule;

    // constraint scaling

    // filter
    double *filter;
    int filter_size;
    int filter_capacity;
    int policy_valid;           // 1: the update rules and the iterate in nlp_out are those of the last solve, from
                                // which ocp_nlp_filterddp_warm_start_from_policy rolls out

    // iteration data
    double mu;
    double reg_last;
    double step_size;
    double objective;
    double primal_inf;          // max(eq_inf, ineq_inf)
    double eq_inf;              // equality constraint violation
    double ineq_inf;            // inequality constraint violation (slack residual)
    double dual_inf;
    int stationarity_costate;   // 1: dual_inf is attained with the costate, 0: with the primal-dual value gradient
    double cs_inf_0;
    double cs_inf_mu;
    double barrier_lagrangian_curr;
    double primal_1_curr;
    double barrier_lagrangian_next;
    double primal_1_next;
    double expected_change_L;
    double theta_max;
    double theta_min;
    int switching;
    int armijo_passed;
    int barrier_iter;
    int line_search_iter;
    int status_internal;
    int warm_started;           // 1 if the last solve started from the shifted update rules
    double policy_gamma;        // feedforward factor of the update rules left for the warm start
    int warm_rows_fresh;        // constraint rows the shift had no previous rule for, initialized as in a cold start

    int ni_bounds;

    // timeout memory
    double timeout_estimated_per_iteration_time;

} ocp_nlp_filterddp_memory;

//
acados_size_t ocp_nlp_filterddp_memory_calculate_size(void *config, void *dims, void *opts_, void *in_);
//
void *ocp_nlp_filterddp_memory_assign(void *config, void *dims, void *opts_, void *in_, void *raw_memory);
//
void ocp_nlp_filterddp_memory_reset_qp_solver(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
    void *opts_, void *mem_, void *work_);



/************************************************
 * workspace
 ************************************************/

typedef struct
{
    ocp_nlp_workspace *nlp_work;

    // value function
    struct blasfeo_dvec Vx;
    struct blasfeo_dvec Vx_next;
    struct blasfeo_dmat Vxx;
    struct blasfeo_dvec lambda;
    struct blasfeo_dvec lambda_next;
    struct blasfeo_dvec Vd;       // primal-dual value gradient, multiplier estimate of the stationarity measure
    struct blasfeo_dvec Vd_next;

    // stage derivatives gathered from the qp_in linearization
    struct blasfeo_dmat fx;
    struct blasfeo_dmat fu;
    struct blasfeo_dmat hx;
    struct blasfeo_dmat hu;
    struct blasfeo_dmat gx;
    struct blasfeo_dmat gu;
    struct blasfeo_dvec lx;
    struct blasfeo_dvec lu;
    struct blasfeo_dvec h;
    struct blasfeo_dvec g;
    struct blasfeo_dvec q;

    // stage Q function
    struct blasfeo_dmat C;
    struct blasfeo_dmat H;
    struct blasfeo_dmat B;
    struct blasfeo_dmat xx_tmp;
    struct blasfeo_dmat ux_tmp;
    struct blasfeo_dmat betaY;    // range space part Y aby of the feedback gain (symmetric value Hessian form)
    struct blasfeo_dmat HbetaY;   // H betaY
    struct blasfeo_dmat sg_tmp;   // Sigmas gx
    struct blasfeo_dvec pi_tmp;   // elementwise selection of Vx and lambda
    struct blasfeo_dvec Qu;
    struct blasfeo_dvec Qs;
    struct blasfeo_dvec Lu;
    struct blasfeo_dvec Lu_costate;
    struct blasfeo_dvec Ls;

    // bound terms
    struct blasfeo_dvec ul_dist;
    struct blasfeo_dvec uu_dist;
    struct blasfeo_dvec inv_ul;
    struct blasfeo_dvec inv_uu;
    struct blasfeo_dvec SigmaL;
    struct blasfeo_dvec SigmaU;
    struct blasfeo_dvec sl_dist;
    struct blasfeo_dvec su_dist;
    struct blasfeo_dvec inv_sl;
    struct blasfeo_dvec inv_su;
    struct blasfeo_dvec SigmasL;
    struct blasfeo_dvec SigmasU;
    struct blasfeo_dvec Sigmas;

    // soft constraint terms per inequality row, zero for hard sides
    struct blasfeo_dvec inv_el;   // 1/(sigl - lsl)
    struct blasfeo_dvec inv_eu;
    struct blasfeo_dvec Xil;      // xil/(sigl - lsl)
    struct blasfeo_dvec Xiu;
    struct blasfeo_dvec Qsigl;    // barrier gradient with respect to the slack
    struct blasfeo_dvec Qsigu;
    struct blasfeo_dvec rsigl;    // stationarity residual of the slack
    struct blasfeo_dvec rsigu;
    struct blasfeo_dvec inv_Dl;   // 1/(Zl + SigmasL + Xil)
    struct blasfeo_dvec inv_Du;
    struct blasfeo_dvec ksigl;    // SigmasL/Dl, feedback of the slack on s
    struct blasfeo_dvec ksigu;
    struct blasfeo_dvec csigl;    // SigmasL Qsigl/Dl
    struct blasfeo_dvec csigu;
    struct blasfeo_dvec wsl;      // (Zl + Xil)/Dl, factor of SigmasL after the elimination
    struct blasfeo_dvec wsu;

    // stage KKT system
    struct blasfeo_dmat rhs_u;
    struct blasfeo_dmat rhs_s;
    struct blasfeo_dmat rhs_g;
    struct blasfeo_dmat Hsolve;
    struct blasfeo_dmat rhs_u_solve;
    struct blasfeo_dmat guT;
    struct blasfeo_dmat guT_S;
    struct blasfeo_dmat rhs_sg;
    struct blasfeo_dmat Lchol;
    struct blasfeo_dmat sol_tmp;

    // null space method
    struct blasfeo_dmat lq;
    struct blasfeo_dmat Q;
    struct blasfeo_dmat Y;
    struct blasfeo_dmat Z;
    struct blasfeo_dmat AY;
    struct blasfeo_dmat AY_lu;
    struct blasfeo_dmat aby;
    struct blasfeo_dmat aby_tmp;
    struct blasfeo_dmat HY;
    struct blasfeo_dmat ZH;
    struct blasfeo_dmat M;
    struct blasfeo_dmat LM;
    struct blasfeo_dmat tmp_u;
    struct blasfeo_dmat abz;
    struct blasfeo_dmat abz_tmp;
    struct blasfeo_dmat yr;
    struct blasfeo_dmat yr_tmp;
    int *ipiv;
    void *lq_work;
    void *orglq_work;

    // rollout
    struct blasfeo_dvec xi;
    struct blasfeo_dvec tmp_nv;
    struct blasfeo_dvec tmp_2ni;

} ocp_nlp_filterddp_workspace;

//
acados_size_t ocp_nlp_filterddp_workspace_calculate_size(void *config, void *dims, void *opts_, void *in_);



/************************************************
 * functions
 ************************************************/

//
int ocp_nlp_filterddp(void *config, void *dims, void *nlp_in, void *nlp_out,
                void *args, void *mem, void *work_);
//
void ocp_nlp_filterddp_config_initialize_default(void *config_);
//
int ocp_nlp_filterddp_precompute(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_);
//
void ocp_nlp_filterddp_get(void *config_, void *dims_, void *mem_, const char *field, void *return_value_);
//
void ocp_nlp_filterddp_get_at_stage(void *config_, void *dims_, void *mem_, int stage, const char *field, void *return_value_);
//
int ocp_nlp_filterddp_warm_start_from_policy(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_, double *x0);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_FILTERDDP_H_
/// @}
/// @}
/// @}
