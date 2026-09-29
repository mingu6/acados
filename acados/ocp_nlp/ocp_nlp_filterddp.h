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

    double mu_init;             // barrier parameter initialization (scaled by objective scaling)
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

    int nlp_scaling;            // gradient based scaling of objective and constraints at the initial point
    double nlp_scaling_max_gradient;
    int warm_start;             // 1: initialize each solve from the shifted affine policy and final barrier parameter of the previous solve
    int symmetric_value_hessian; // 1: symmetrise the value function Hessian after each stage

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

    // bounds gathered per stage in solver order
    struct blasfeo_dvec *ul;    // control lower bounds
    struct blasfeo_dvec *uu;    // control upper bounds
    struct blasfeo_dvec *maskul;
    struct blasfeo_dvec *maskuu;
    struct blasfeo_dvec *gl;    // slack lower bounds
    struct blasfeo_dvec *gu;    // slack upper bounds
    struct blasfeo_dvec *maskgl;
    struct blasfeo_dvec *maskgu;

    // iterate: nlp_out holds x, u; dual variables and slacks are solver specific
    struct blasfeo_dvec *s;     // inequality slacks
    struct blasfeo_dvec *phi;   // equality multipliers
    struct blasfeo_dvec *nu;    // inequality multipliers
    struct blasfeo_dvec *zl;    // control lower bound multipliers
    struct blasfeo_dvec *zu;    // control upper bound multipliers
    struct blasfeo_dvec *zsl;   // slack lower bound multipliers
    struct blasfeo_dvec *zsu;   // slack upper bound multipliers
    // trial iterate
    struct blasfeo_dvec *s_trial;
    struct blasfeo_dvec *phi_trial;
    struct blasfeo_dvec *nu_trial;
    struct blasfeo_dvec *zl_trial;
    struct blasfeo_dvec *zu_trial;
    struct blasfeo_dvec *zsl_trial;
    struct blasfeo_dvec *zsu_trial;

    // update rules, feedforward in column 0, feedback in columns 1:nx
    struct blasfeo_dmat *alpha_beta;      // nu x (nx+1)
    struct blasfeo_dmat *alphas_betas;    // ng x (nx+1)
    struct blasfeo_dmat *psih_omegah;     // nh x (nx+1)
    struct blasfeo_dmat *psig_omegag;     // ng x (nx+1)
    struct blasfeo_dmat *chil_zetal;      // nu x (nx+1)
    struct blasfeo_dmat *chiu_zetau;      // nu x (nx+1)
    struct blasfeo_dmat *chisl_zetasl;    // ng x (nx+1)
    struct blasfeo_dmat *chisu_zetasu;    // ng x (nx+1)

    // constraint scaling
    struct blasfeo_dvec *h_scale;
    struct blasfeo_dvec *g_scale;
    double objective_scale;

    // filter
    double *filter;
    int filter_size;
    int filter_capacity;
    int policy_valid;

    // iteration data
    double mu;
    double reg_last;
    double step_size;
    double objective;
    double primal_inf;
    double primal_inf_raw;
    double dual_inf;
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

    int ni_bounds;

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
    struct blasfeo_dvec h_scaled;
    struct blasfeo_dvec g;
    struct blasfeo_dvec q;

    // stage Q function
    struct blasfeo_dmat C;
    struct blasfeo_dmat H;
    struct blasfeo_dmat B;
    struct blasfeo_dmat xx_tmp;
    struct blasfeo_dmat ux_tmp;
    struct blasfeo_dvec Qu;
    struct blasfeo_dvec Qs;
    struct blasfeo_dvec Lu;
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
    struct blasfeo_dmat hus;
    struct blasfeo_dmat hxs;
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_FILTERDDP_H_
/// @}
/// @}
/// @}
