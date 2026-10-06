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


#include "acados/ocp_nlp/ocp_nlp_filterddp.h"

// external
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#if defined(ACADOS_WITH_OPENMP)
#include <omp.h>
#endif

// blasfeo
#include "blasfeo_d_aux.h"
#include "blasfeo_d_aux_ext_dep.h"
#include "blasfeo_d_blas.h"
// acados
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"
#include "acados/utils/strsep.h"
#include "acados_c/ocp_qp_interface.h"

#define EL BLASFEO_DMATEL
#define VEL BLASFEO_DVECEL

#define FILTERDDP_STATUS_OK 0
#define FILTERDDP_STATUS_BACKWARD_PASS_FAILED 1
#define FILTERDDP_STATUS_FRACTION_TO_BOUNDARY 2
#define FILTERDDP_STATUS_FILTER_BLOCKED 3
#define FILTERDDP_STATUS_STEP_ACCEPTANCE 4
#define FILTERDDP_STATUS_FORWARD_FAILED 5
#define FILTERDDP_STATUS_LINE_SEARCH_FAILED 7

/************************************************
 * options
 ************************************************/

acados_size_t ocp_nlp_filterddp_opts_calculate_size(void *config_, void *dims_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_filterddp_opts);

    size += ocp_nlp_opts_calculate_size(config, dims);

    return size;
}



void *ocp_nlp_filterddp_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;

    char *c_ptr = (char *) raw_memory;

    ocp_nlp_filterddp_opts *opts = (ocp_nlp_filterddp_opts *) c_ptr;
    c_ptr += sizeof(ocp_nlp_filterddp_opts);

    opts->nlp_opts = ocp_nlp_opts_assign(config, dims, c_ptr);
    c_ptr += ocp_nlp_opts_calculate_size(config, dims);

    assert((char *) raw_memory + ocp_nlp_filterddp_opts_calculate_size(config, dims) >= c_ptr);

    return opts;
}



void ocp_nlp_filterddp_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    // this first !!!
    ocp_nlp_opts_initialize_default(config, dims, nlp_opts);

    nlp_opts->max_iter = 1000;

    opts->mu_init = 1.0;
    opts->ineq_dual_init = 1.0;
    opts->kappa_1 = 0.01;
    opts->kappa_2 = 0.01;

    opts->reg_1 = 1e-4;
    opts->reg_min = 1e-20;
    opts->reg_max = 1e40;
    opts->kappa_bar_w_p = 100.0;
    opts->kappa_w_p = 8.0;
    opts->kappa_w_m = 1.0 / 3.0;

    opts->kappa_eps = 10.0;
    opts->kappa_mu = 0.2;
    opts->theta_mu = 1.2;
    opts->tau_min = 0.99;

    opts->s_max = 100.0;
    opts->eta_L = 1e-4;
    opts->s_L = 2.3;
    opts->delta = 1.0;
    opts->s_theta = 1.1;
    opts->gamma_theta = 1e-5;
    opts->gamma_L = 1e-5;
    opts->theta_max_factor = 1e6;
    opts->theta_min_factor = 1e-4;

    opts->warm_start = 0;
    opts->bound_mult_init_method = 0;
    opts->policy_at_cap = 1;
    opts->symmetric_value_hessian = 2;
    opts->dynamics_multiplier = 0;

    opts->timeout_heuristic = ZERO;
    opts->timeout_max_time = 0; // corresponds to no timeout

    return;
}



void ocp_nlp_filterddp_opts_update(void *config_, void *dims_, void *opts_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    ocp_nlp_opts_update(config, dims, nlp_opts);

    return;
}



void ocp_nlp_filterddp_opts_set(void *config_, void *opts_, const char *field, void* value)
{
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = (ocp_nlp_filterddp_opts *) opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    char *ptr_module = NULL;
    int module_length = 0;
    char module[MAX_STR_LEN];
    extract_module_name(field, module, &module_length, &ptr_module);

    if ( ptr_module!=NULL && (!strcmp(ptr_module, "qp")) )
    {
        ocp_nlp_opts_set(config, nlp_opts, field, value);
    }
    else if (!strcmp(field, "filterddp_mu_init"))
    {
        opts->mu_init = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_ineq_dual_init"))
    {
        opts->ineq_dual_init = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_1"))
    {
        opts->kappa_1 = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_2"))
    {
        opts->kappa_2 = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_reg_1"))
    {
        opts->reg_1 = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_reg_min"))
    {
        opts->reg_min = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_reg_max"))
    {
        opts->reg_max = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_bar_w_p"))
    {
        opts->kappa_bar_w_p = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_w_p"))
    {
        opts->kappa_w_p = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_w_m"))
    {
        opts->kappa_w_m = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_eps"))
    {
        opts->kappa_eps = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_kappa_mu"))
    {
        opts->kappa_mu = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_theta_mu"))
    {
        opts->theta_mu = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_tau_min"))
    {
        opts->tau_min = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_s_max"))
    {
        opts->s_max = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_eta_L"))
    {
        opts->eta_L = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_s_L"))
    {
        opts->s_L = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_delta"))
    {
        opts->delta = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_s_theta"))
    {
        opts->s_theta = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_gamma_theta"))
    {
        opts->gamma_theta = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_gamma_L"))
    {
        opts->gamma_L = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_theta_max_factor"))
    {
        opts->theta_max_factor = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_theta_min_factor"))
    {
        opts->theta_min_factor = *(double *) value;
    }
    else if (!strcmp(field, "filterddp_warm_start"))
    {
        opts->warm_start = *(int *) value;
    }
    else if (!strcmp(field, "filterddp_bound_mult_init_method"))
    {
        int method = *(int *) value;
        if (method != 0 && method != 1)
        {
            printf("\nerror: ocp_nlp_filterddp_opts_set: filterddp_bound_mult_init_method must be 0 (constant) or 1 (mu_based), got %d.\n", method);
            exit(1);
        }
        opts->bound_mult_init_method = method;
    }
    else if (!strcmp(field, "filterddp_policy_at_cap"))
    {
        opts->policy_at_cap = *(int *) value;
    }
    else if (!strcmp(field, "filterddp_symmetric_value_hessian"))
    {
        opts->symmetric_value_hessian = *(int *) value;
    }
    else if (!strcmp(field, "filterddp_dynamics_multiplier"))
    {
        opts->dynamics_multiplier = *(int *) value;
    }
    else if (!strcmp(field, "timeout_max_time"))
    {
        opts->timeout_max_time = *(double *) value;
    }
    else if (!strcmp(field, "timeout_heuristic"))
    {
        opts->timeout_heuristic = *(ocp_nlp_timeout_heuristic_t *) value;
    }
    else
    {
        ocp_nlp_opts_set(config, nlp_opts, field, value);
    }

    return;
}



void ocp_nlp_filterddp_opts_set_at_stage(void *config_, void *opts_, size_t stage, const char *field, void* value)
{
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = (ocp_nlp_filterddp_opts *) opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    ocp_nlp_opts_set_at_stage(config, nlp_opts, stage, field, value);

    return;
}



/************************************************
 * memory
 ************************************************/

static void filterddp_dims_max(ocp_nlp_dims *dims, int *nx_max, int *nu_max, int *ni_max)
{
    int N = dims->N;
    *nx_max = 0;
    *nu_max = 0;
    *ni_max = 0;
    for (int i = 0; i <= N; i++)
    {
        *nx_max = *nx_max > dims->nx[i] ? *nx_max : dims->nx[i];
        *nu_max = *nu_max > dims->nu[i] ? *nu_max : dims->nu[i];
        *ni_max = *ni_max > dims->ni[i] ? *ni_max : dims->ni[i];
    }
}



acados_size_t ocp_nlp_filterddp_memory_calculate_size(void *config_, void *dims_, void *opts_, void *in_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_in *in = in_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ni = dims->ni;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_filterddp_memory);

    // nlp mem
    size += ocp_nlp_memory_calculate_size(config, dims, nlp_opts, in);

    // stat
    int stat_m = nlp_opts->max_iter+1;
    int stat_n = 8;
    size += stat_n*stat_m*sizeof(double);

    // classification, and that of the previous solve
    size += 8*N*sizeof(int);
    size += 8*N*sizeof(int *);
    for (int i = 0; i < N; i++)
    {
        size += 8*ni[i]*sizeof(int);
    }

    // per stage vectors: 12 bounds, 9 iterate, 9 trial, 2 scaling
    size += 32*N*sizeof(struct blasfeo_dvec);
    // costate
    size += (N+1)*sizeof(struct blasfeo_dvec);
    for (int i = 0; i <= N; i++)
        size += blasfeo_memsize_dvec(nx[i]);
    // per stage matrices: 12 update rules
    size += 12*N*sizeof(struct blasfeo_dmat);
    for (int i = 0; i < N; i++)
    {
        size += 4*blasfeo_memsize_dvec(nu[i]);         // ul, uu, maskul, maskuu
        size += 4*blasfeo_memsize_dvec(ni[i]);         // gl, gu, maskgl, maskgu
        size += 2*blasfeo_memsize_dvec(ni[i]);         // s, s_trial
        size += 2*blasfeo_memsize_dvec(ni[i]);         // phi, phi_trial
        size += 2*blasfeo_memsize_dvec(ni[i]);         // nu, nu_trial
        size += 4*blasfeo_memsize_dvec(nu[i]);         // zl, zu, zl_trial, zu_trial
        size += 4*blasfeo_memsize_dvec(ni[i]);         // zsl, zsu, zsl_trial, zsu_trial
        size += 8*blasfeo_memsize_dvec(ni[i]);         // lsl, lsu, masksl, masksu, xil, xiu, xil_trial, xiu_trial
        size += 4*blasfeo_memsize_dmat(ni[i], nx[i]+1);   // sigl_rule, sigu_rule, xil_rule, xiu_rule
        size += 4*blasfeo_memsize_dmat(nu[i], nx[i]+1);   // alpha_beta, chil_zetal, chiu_zetau + spare
        size += 4*blasfeo_memsize_dmat(ni[i], nx[i]+1);   // alphas_betas, psih_omegah, psig_omegag, chisl_zetasl, chisu_zetasu
        size += blasfeo_memsize_dmat(ni[i], nx[i]+1);
    }

    // filter
    size += 2*(nlp_opts->max_iter+2)*sizeof(double);

    size += 3*8;
    size += 64;

    make_int_multiple_of(8, &size);

    return size;
}



void *ocp_nlp_filterddp_memory_assign(void *config_, void *dims_, void *opts_, void *in_, void *raw_memory)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_in *in = in_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;

    char *c_ptr = (char *) raw_memory;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ni = dims->ni;

    // initial align
    align_char_to(8, &c_ptr);

    ocp_nlp_filterddp_memory *mem = (ocp_nlp_filterddp_memory *) c_ptr;
    c_ptr += sizeof(ocp_nlp_filterddp_memory);

    align_char_to(8, &c_ptr);

    // nlp mem
    mem->nlp_mem = ocp_nlp_memory_assign(config, dims, nlp_opts, in, c_ptr);
    c_ptr += ocp_nlp_memory_calculate_size(config, dims, nlp_opts, in);

    // stat
    mem->stat = (double *) c_ptr;
    mem->stat_m = nlp_opts->max_iter+1;
    mem->stat_n = 8;
    c_ptr += mem->stat_m*mem->stat_n*sizeof(double);

    // filter
    mem->filter = (double *) c_ptr;
    mem->filter_capacity = nlp_opts->max_iter+2;
    c_ptr += 2*mem->filter_capacity*sizeof(double);

    // classification
    assign_and_advance_int(N, &mem->nul, &c_ptr);
    assign_and_advance_int(N, &mem->nuu, &c_ptr);
    assign_and_advance_int(N, &mem->nh, &c_ptr);
    assign_and_advance_int(N, &mem->ng, &c_ptr);
    assign_and_advance_int(N, &mem->ngl, &c_ptr);
    assign_and_advance_int(N, &mem->ngu, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxh, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxg, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxs_row, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxs_g, &c_ptr);
    assign_and_advance_int(N, &mem->nh_prev, &c_ptr);
    assign_and_advance_int(N, &mem->ng_prev, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxh_prev, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxg_prev, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->idxs_g_prev, &c_ptr);
    assign_and_advance_int_ptrs(N, &mem->sides_prev, &c_ptr);
    for (int i = 0; i < N; i++)
    {
        assign_and_advance_int(ni[i], &mem->idxh[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxg[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxs_row[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxs_g[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxh_prev[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxg_prev[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxs_g_prev[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->sides_prev[i], &c_ptr);
    }

    // vector structs
    assign_and_advance_blasfeo_dvec_structs(N, &mem->ul, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->uu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->maskul, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->maskuu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->gl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->gu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->maskgl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->maskgu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->s, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->phi, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->nu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zsl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zsu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->s_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->phi_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->nu_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zl_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zu_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zsl_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->zsu_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->lsl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->lsu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->masksl, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->masksu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->xil, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->xiu, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->xil_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->xiu_trial, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N+1, &mem->costate, &c_ptr);

    // matrix structs
    assign_and_advance_blasfeo_dmat_structs(N, &mem->alpha_beta, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->alphas_betas, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->psih_omegah, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->psig_omegag, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chil_zetal, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chiu_zetau, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chisl_zetasl, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chisu_zetasu, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->sigl_rule, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->sigu_rule, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->xil_rule, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->xiu_rule, &c_ptr);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);

    for (int i = 0; i < N; i++)
    {
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->ul+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->uu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->maskul+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->maskuu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->gl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->gu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->maskgl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->maskgu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->s+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->phi+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->nu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->zl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->zu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->zsl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->zsu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->s_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->phi_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->nu_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->zl_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(nu[i], mem->zu_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->zsl_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->zsu_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->lsl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->lsu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->masksl+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->masksu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->xil+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->xiu+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->xil_trial+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->xiu_trial+i, &c_ptr);

        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->alpha_beta+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->alphas_betas+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->psih_omegah+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->psig_omegag+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->chil_zetal+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->chiu_zetau+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->chisl_zetasl+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->chisu_zetasu+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->sigl_rule+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->sigu_rule+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->xil_rule+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->xiu_rule+i, &c_ptr);
    }
    for (int i = 0; i <= N; i++)
        assign_and_advance_blasfeo_dvec_mem(nx[i], mem->costate+i, &c_ptr);

    for (int i = 0; i < N; i++)
    {
        blasfeo_dvecse(ni[i], 0.0, mem->s+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->phi+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->nu+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->xil+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->xiu+i, 0);
        blasfeo_dvecse(nu[i], 0.0, mem->zl+i, 0);
        blasfeo_dvecse(nu[i], 0.0, mem->zu+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->zsl+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->zsu+i, 0);
    }

    mem->filter_size = 0;
    mem->policy_valid = 0;
    mem->duals_valid = 0;
    mem->timeout_estimated_per_iteration_time = 0;
    mem->warm_started = 0;
    mem->warm_rows_fresh = 0;
    mem->policy_gamma = 1.0;
    mem->nlp_mem->status = ACADOS_READY;

    align_char_to(8, &c_ptr);

    assert((char *) raw_memory + ocp_nlp_filterddp_memory_calculate_size(config, dims, opts, in) >= c_ptr);

    return mem;
}



/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_nlp_filterddp_workspace_calculate_size(void *config_, void *dims_, void *opts_, void *in_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_in *nlp_in = in_;

    int nx_max, nu_max, ni_max;
    filterddp_dims_max(dims, &nx_max, &nu_max, &ni_max);
    int nxp = nx_max+1;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_filterddp_workspace);

    size += ocp_nlp_workspace_calculate_size(config, dims, nlp_opts, nlp_in);

    size += 6*blasfeo_memsize_dvec(nx_max);                 // Vx, Vx_next, lambda, lambda_next, Vd, Vd_next
    size += blasfeo_memsize_dmat(nx_max, nx_max);           // Vxx
    size += 2*blasfeo_memsize_dmat(nx_max, nx_max);         // fx, C
    size += blasfeo_memsize_dmat(nx_max, nu_max);           // fu
    size += 3*blasfeo_memsize_dmat(ni_max, nx_max);         // hx, gx + spare
    size += 2*blasfeo_memsize_dmat(ni_max, nu_max);         // hu, gu
    size += 2*blasfeo_memsize_dvec(nx_max);                 // lx + spare
    size += 5*blasfeo_memsize_dvec(ni_max);                 // h, g, q, Ls, Qs
    size += 2*blasfeo_memsize_dvec(nu_max);                 // lu, Qu
    size += 2*blasfeo_memsize_dvec(nu_max);                 // Lu, Lu_costate
    size += 3*blasfeo_memsize_dmat(nu_max, nu_max);         // H, Hsolve, Lchol
    size += 4*blasfeo_memsize_dmat(nu_max, nx_max);         // B, ux_tmp, betaY, HbetaY
    size += blasfeo_memsize_dmat(ni_max, nx_max);           // sg_tmp
    size += blasfeo_memsize_dvec(nx_max);                   // pi_tmp
    size += blasfeo_memsize_dmat(nx_max, nx_max);           // xx_tmp
    size += 6*blasfeo_memsize_dvec(nu_max);                 // ul_dist ... SigmaU
    size += 7*blasfeo_memsize_dvec(ni_max);                 // sl_dist ... Sigmas
    size += 16*blasfeo_memsize_dvec(ni_max);                // inv_el ... wsu
    size += 4*blasfeo_memsize_dmat(nu_max, nxp);            // rhs_u, rhs_u_solve, sol_tmp, tmp_u
    size += 3*blasfeo_memsize_dmat(ni_max, nxp);            // rhs_s, rhs_g, rhs_sg
    size += 2*blasfeo_memsize_dmat(nu_max, ni_max);         // guT, guT_S
    size += blasfeo_memsize_dmat(ni_max, nu_max);           // lq
    size += blasfeo_memsize_dmat(nu_max, nu_max);           // Q
    size += 2*blasfeo_memsize_dmat(nu_max, nu_max);         // Y, Z
    size += 2*blasfeo_memsize_dmat(ni_max, ni_max);         // AY, AY_lu
    size += 4*blasfeo_memsize_dmat(ni_max, nxp);            // aby, aby_tmp, yr, yr_tmp
    size += blasfeo_memsize_dmat(nu_max, ni_max);           // HY
    size += blasfeo_memsize_dmat(nu_max, nu_max);           // ZH
    size += 2*blasfeo_memsize_dmat(nu_max, nu_max);         // M, LM
    size += 2*blasfeo_memsize_dmat(nu_max, nxp);            // abz, abz_tmp
    size += ni_max*sizeof(int);                             // ipiv
    size += blasfeo_dgelqf_worksize(ni_max, nu_max);
    size += blasfeo_dorglq_worksize(nu_max, nu_max, ni_max);
    size += blasfeo_memsize_dvec(nxp);                      // xi
    size += blasfeo_memsize_dvec(nu_max+nx_max);            // tmp_nv
    size += blasfeo_memsize_dvec(2*ni_max);                 // tmp_2ni

    size += 3*64;
    size += 8;

    make_int_multiple_of(8, &size);

    return size;
}



static void ocp_nlp_filterddp_cast_workspace(ocp_nlp_config *config, ocp_nlp_dims *dims,
         ocp_nlp_filterddp_opts *opts, ocp_nlp_in *nlp_in, ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;

    int nx_max, nu_max, ni_max;
    filterddp_dims_max(dims, &nx_max, &nu_max, &ni_max);
    int nxp = nx_max+1;

    char *c_ptr = (char *) work;
    c_ptr += sizeof(ocp_nlp_filterddp_workspace);

    // nlp
    work->nlp_work = ocp_nlp_workspace_assign(config, dims, nlp_opts, nlp_in, nlp_mem, c_ptr);
    c_ptr += ocp_nlp_workspace_calculate_size(config, dims, nlp_opts, nlp_in);

    align_char_to(64, &c_ptr);

    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->Vx, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->Vx_next, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->Vxx, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->lambda, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->lambda_next, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->Vd, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->Vd_next, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->fx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nx_max, nu_max, &work->fu, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->hx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->hu, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->gx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->gu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->lx, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->lu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->h, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->g, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->q, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->C, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->H, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->B, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->xx_tmp, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->ux_tmp, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->betaY, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->HbetaY, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->sg_tmp, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->pi_tmp, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->Qu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Qs, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->Lu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->Lu_costate, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Ls, &c_ptr);

    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->ul_dist, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->uu_dist, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->inv_ul, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->inv_uu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->SigmaL, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->SigmaU, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->sl_dist, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->su_dist, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_sl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_su, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->SigmasL, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->SigmasU, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Sigmas, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_el, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_eu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Xil, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Xiu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Qsigl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Qsigu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->rsigl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->rsigu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_Dl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->inv_Du, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->ksigl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->ksigu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->csigl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->csigu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->wsl, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->wsu, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->rhs_u, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->rhs_s, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->rhs_g, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->Hsolve, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->rhs_u_solve, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, ni_max, &work->guT, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, ni_max, &work->guT_S, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->rhs_sg, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->Lchol, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->sol_tmp, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->lq, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->Q, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->Y, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->Z, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, ni_max, &work->AY, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, ni_max, &work->AY_lu, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->aby, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->aby_tmp, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, ni_max, &work->HY, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->ZH, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->M, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->LM, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->tmp_u, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->abz, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nxp, &work->abz_tmp, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->yr, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nxp, &work->yr_tmp, &c_ptr);

    assign_and_advance_blasfeo_dvec_mem(nxp, &work->xi, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max+nx_max, &work->tmp_nv, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(2*ni_max, &work->tmp_2ni, &c_ptr);

    align_char_to(64, &c_ptr);
    work->lq_work = (void *) c_ptr;
    c_ptr += blasfeo_dgelqf_worksize(ni_max, nu_max);
    align_char_to(64, &c_ptr);
    work->orglq_work = (void *) c_ptr;
    c_ptr += blasfeo_dorglq_worksize(nu_max, nu_max, ni_max);

    assign_and_advance_int(ni_max, &work->ipiv, &c_ptr);

    assert((char *) work + ocp_nlp_filterddp_workspace_calculate_size(config, dims, opts, nlp_in) >= c_ptr);

    return;
}



/************************************************
 * helper functions
 ************************************************/

static double filterddp_max(double a, double b)
{
    return (a > b || isnan(a)) ? a : b;
}



static double filterddp_min(double a, double b)
{
    return (a < b || isnan(a)) ? a : b;
}



static int filterddp_cholesky_ok(int n, struct blasfeo_dmat *L)
{
    for (int i = 0; i < n; i++)
    {
        if (!(EL(L, i, i) > 0.0))
            return 0;
    }
    return 1;
}



static void filterddp_symmetrize_from_upper(int n, struct blasfeo_dmat *A)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = j+1; i < n; i++)
        {
            EL(A, i, j) = EL(A, j, i);
        }
    }
}



static int filterddp_all_finite(int n, struct blasfeo_dvec *v, int vi)
{
    for (int i = 0; i < n; i++)
    {
        if (!isfinite(VEL(v, vi+i)))
            return 0;
    }
    return 1;
}



static double filterddp_norm_inf(int n, struct blasfeo_dvec *v, int vi)
{
    double r = 0.0;
    for (int i = 0; i < n; i++)
    {
        r = filterddp_max(r, fabs(VEL(v, vi+i)));
    }
    return r;
}



static double filterddp_norm_1(int n, struct blasfeo_dvec *v, int vi)
{
    double r = 0.0;
    for (int i = 0; i < n; i++)
    {
        r += fabs(VEL(v, vi+i));
    }
    return r;
}



static double filterddp_interior(double value, double lower, double upper, int has_lower, int has_upper,
        double kappa_1, double kappa_2)
{
    if (has_lower && has_upper)
    {
        double lower_interior = lower + filterddp_min(kappa_1*filterddp_max(1.0, fabs(lower)), kappa_2*(upper-lower));
        double upper_interior = upper - filterddp_min(kappa_1*filterddp_max(1.0, fabs(upper)), kappa_2*(upper-lower));
        return filterddp_min(filterddp_max(value, lower_interior), upper_interior);
    }
    else if (has_lower)
    {
        return filterddp_max(value, kappa_1*filterddp_max(lower, 1.0) + lower);
    }
    else if (has_upper)
    {
        return filterddp_min(value, -kappa_1*filterddp_max(upper, 1.0) + upper);
    }
    return value;
}



static void filterddp_reset_filter(ocp_nlp_filterddp_memory *mem)
{
    mem->filter[0] = mem->theta_max;
    mem->filter[1] = -INFINITY;
    mem->filter_size = 1;
    mem->status_internal = FILTERDDP_STATUS_OK;
}



static void filterddp_update_filter(ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_opts *opts)
{
    double theta_new = (1.0 - opts->gamma_theta)*mem->primal_1_curr;
    double L_new = mem->barrier_lagrangian_curr - opts->gamma_L*mem->primal_1_curr;
    int next_index = 0;
    for (int i = 0; i < mem->filter_size; i++)
    {
        double theta = mem->filter[2*i];
        double L = mem->filter[2*i+1];
        if (!(theta_new <= theta && L_new <= L))
        {
            mem->filter[2*next_index] = theta;
            mem->filter[2*next_index+1] = L;
            next_index++;
        }
    }
    assert(next_index < mem->filter_capacity);
    mem->filter[2*next_index] = theta_new;
    mem->filter[2*next_index+1] = L_new;
    mem->filter_size = next_index+1;
}



static int filterddp_filter_blocks(ocp_nlp_filterddp_memory *mem, double theta, double L)
{
    for (int i = 0; i < mem->filter_size; i++)
    {
        if (theta >= mem->filter[2*i] && L >= mem->filter[2*i+1])
            return 1;
    }
    return 0;
}



// constraint rows of a stage in the layout [bu; bx; g; h] of the bounds d, the masks and the multipliers
// lam, whose upper sides start at this offset and whose slack bounds follow at twice this offset
static int filterddp_nrows(ocp_nlp_constraints_bgh_dims *cdims)
{
    return cdims->nb + cdims->ng + cdims->nh;
}



// row of an affine update rule evaluated at the state deviation xi
static double filterddp_rule_value(int nx, struct blasfeo_dmat *K, int row, struct blasfeo_dvec *xi, double gamma)
{
    double value = gamma*EL(K, row, 0);
    for (int k = 0; k < nx; k++)
        value += EL(K, row, k+1)*VEL(xi, k);
    return value;
}



/*
 * constraint evaluation of stage i at the point currently aliased as ux in the constraints module:
 * constr_eval_no_bounds = [ux[idxb]; DCt' ux; h(x,u)], gathered into h (equality rows) and g
 * (inequality rows) in solver order.
 */
static void filterddp_gather_constraints(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_opts *nlp_opts, ocp_nlp_memory *nlp_mem, ocp_nlp_workspace *nlp_work, ocp_nlp_filterddp_memory *mem,
        int i, struct blasfeo_dvec *h, struct blasfeo_dvec *g)
{
    ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
    int nh = mem->nh[i];
    int ng = mem->ng[i];

    for (int j = 0; j < nh; j++)
    {
        VEL(h, j) = VEL(&constr_mem->constr_eval_no_bounds, mem->idxh[i][j]);
    }
    for (int j = 0; j < ng; j++)
    {
        VEL(g, j) = VEL(&constr_mem->constr_eval_no_bounds, mem->idxg[i][j]);
    }
}



static void filterddp_gather_jacobians(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_memory *nlp_mem,
        ocp_nlp_filterddp_memory *mem, int i, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
    ocp_nlp_constraints_bgh_dims *constr_dims = dims->constraints[i];
    ocp_qp_in *qp_in = nlp_mem->qp_in;
    int nx = dims->nx[i];
    int nu = dims->nu[i];
    int nb = constr_dims->nb;
    int nh = mem->nh[i];
    int ng = mem->ng[i];

    for (int j = 0; j < nh; j++)
    {
        int idx = mem->idxh[i][j];
        if (idx < nb)
        {
            int col = constr_mem->idxb[idx];
            blasfeo_dgese(1, nu, 0.0, &work->hu, j, 0);
            blasfeo_dgese(1, nx, 0.0, &work->hx, j, 0);
            if (col < nu)
                EL(&work->hu, j, col) = 1.0;
            else
                EL(&work->hx, j, col-nu) = 1.0;
        }
        else
        {
            int col = idx-nb;
            for (int k = 0; k < nu; k++)
                EL(&work->hu, j, k) = EL(qp_in->DCt+i, k, col);
            for (int k = 0; k < nx; k++)
                EL(&work->hx, j, k) = EL(qp_in->DCt+i, nu+k, col);
        }
    }
    for (int j = 0; j < ng; j++)
    {
        int idx = mem->idxg[i][j];
        if (idx < nb)
        {
            int col = constr_mem->idxb[idx];
            blasfeo_dgese(1, nu, 0.0, &work->gu, j, 0);
            blasfeo_dgese(1, nx, 0.0, &work->gx, j, 0);
            if (col < nu)
                EL(&work->gu, j, col) = 1.0;
            else
                EL(&work->gx, j, col-nu) = 1.0;
        }
        else
        {
            int col = idx-nb;
            for (int k = 0; k < nu; k++)
                EL(&work->gu, j, k) = EL(qp_in->DCt+i, k, col);
            for (int k = 0; k < nx; k++)
                EL(&work->gx, j, k) = EL(qp_in->DCt+i, nu+k, col);
        }
    }
}



/*
 * The constraints module weights its Hessian contribution with lam_upper - lam_lower.
 * Equality rows carry phi, inequality rows carry nu, in the lam vector of nlp_out.
 */
static void filterddp_set_lam(ocp_nlp_dims *dims, ocp_nlp_out *out, ocp_nlp_filterddp_memory *mem, int i,
        struct blasfeo_dvec *phi, struct blasfeo_dvec *nu_)
{
    int ni0 = filterddp_nrows(dims->constraints[i]);
    blasfeo_dvecse(2*dims->ni[i], 0.0, out->lam+i, 0);
    for (int j = 0; j < mem->nh[i]; j++)
    {
        double m = VEL(phi, j);
        if (m >= 0.0)
            VEL(out->lam+i, ni0+mem->idxh[i][j]) = m;
        else
            VEL(out->lam+i, mem->idxh[i][j]) = -m;
    }
    for (int j = 0; j < mem->ng[i]; j++)
    {
        double m = VEL(nu_, j);
        if (m >= 0.0)
            VEL(out->lam+i, ni0+mem->idxg[i][j]) = m;
        else
            VEL(out->lam+i, mem->idxg[i][j]) = -m;
    }
}



// lower bound on the barrier parameter, as in IPOPT: an order of magnitude below the stationarity and
// complementarity tolerances
static double filterddp_mu_min(ocp_nlp_opts *nlp_opts)
{
    return filterddp_min(nlp_opts->tol_stat, nlp_opts->tol_comp)/10.0;
}



static void filterddp_set_pi(ocp_nlp_dims *dims, ocp_nlp_out *out, int i, struct blasfeo_dvec *lambda)
{
    blasfeo_dveccp(dims->nx[i+1], lambda, 0, out->pi+i, 0);
}



/*
 * Classify the constraint rows of each stage: hard bounds on u are handled as control limits, the initial
 * state bound fixes x_0, all other rows are equalities if both sides coincide and inequalities with a
 * slack otherwise. Soft rows are inequalities whose slack bounds are relaxed by the soft constraint
 * slacks, one per row. Bounds are user data that may change between solves, so this runs at every solve.
 */
static int filterddp_classify_constraints(ocp_nlp_dims *dims, ocp_nlp_in *nlp_in, ocp_nlp_filterddp_memory *mem)
{
    int N = dims->N;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_bgh_model *model = nlp_in->constraints[i];
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        int ni0 = filterddp_nrows(cdims);
        int nbu = cdims->nbu;
        int nbx = cdims->nbx;
        int *row_slack = mem->idxs_row[i];
        for (int j = 0; j < ni0; j++)
            row_slack[j] = -1;
        if (model->use_idxs_rev)
        {
            for (int j = 0; j < ni0; j++)
                row_slack[j] = model->idxs_rev[j];
        }
        else
        {
            for (int j = 0; j < cdims->ns; j++)
                row_slack[model->idxs[j]] = j;
        }
        for (int j = 0; j < ni0; j++)
        {
            for (int k = j+1; k < ni0; k++)
            {
                if (row_slack[j] >= 0 && row_slack[j] == row_slack[k])
                {
                    printf("ocp_nlp_filterddp: stage %d relaxes constraint rows %d and %d with the same slack, which is not supported.\n", i, j, k);
                    return ACADOS_QP_FAILURE;
                }
            }
        }
        mem->nh[i] = 0;
        mem->ng[i] = 0;
        for (int j = 0; j < ni0; j++)
        {
            int soft = row_slack[j] >= 0;
            if (j < nbu && !soft)
                continue;
            if (i == 0 && j >= nbu && j < nbu+nbx)
                continue;
            int has_lower = VEL(nlp_in->dmask+i, j) != 0.0;
            int has_upper = VEL(nlp_in->dmask+i, ni0+j) != 0.0;
            double lower = VEL(&model->d, j);
            double upper = VEL(&model->d, ni0+j);
            if (!soft && has_lower && has_upper && lower == upper)
            {
                mem->idxh[i][mem->nh[i]] = j;
                mem->nh[i]++;
            }
            else if (has_lower || has_upper)
            {
                mem->idxg[i][mem->ng[i]] = j;
                mem->idxs_g[i][mem->ng[i]] = row_slack[j];
                mem->ng[i]++;
            }
        }
        if (mem->nh[i] > dims->nu[i])
        {
            printf("ocp_nlp_filterddp: stage %d has %d equality rows but only %d controls.\n", i, mem->nh[i], dims->nu[i]);
            return ACADOS_QP_FAILURE;
        }
    }
    return ACADOS_SUCCESS;
}



// classification of the constraint rows and their bounds in solver order, from the current bounds of nlp_in
static int filterddp_setup_bounds(ocp_nlp_dims *dims, ocp_nlp_in *nlp_in, ocp_nlp_filterddp_memory *mem)
{
    int status = filterddp_classify_constraints(dims, nlp_in, mem);
    if (status != ACADOS_SUCCESS)
        return status;

    for (int i = 0; i < dims->N; i++)
    {
        ocp_nlp_constraints_bgh_model *model = nlp_in->constraints[i];
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        int ni0 = filterddp_nrows(cdims);
        blasfeo_dvecse(dims->nu[i], -INFINITY, mem->ul+i, 0);
        blasfeo_dvecse(dims->nu[i], INFINITY, mem->uu+i, 0);
        blasfeo_dvecse(dims->nu[i], 0.0, mem->maskul+i, 0);
        blasfeo_dvecse(dims->nu[i], 0.0, mem->maskuu+i, 0);
        for (int j = 0; j < cdims->nbu; j++)
        {
            int col = model->idxb[j];
            if (mem->idxs_row[i][j] >= 0)
                continue; // soft control bounds are inequality rows
            if (VEL(nlp_in->dmask+i, j) != 0.0)
            {
                VEL(mem->ul+i, col) = VEL(&model->d, j);
                VEL(mem->maskul+i, col) = 1.0;
            }
            if (VEL(nlp_in->dmask+i, ni0+j) != 0.0)
            {
                VEL(mem->uu+i, col) = VEL(&model->d, ni0+j);
                VEL(mem->maskuu+i, col) = 1.0;
            }
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            int idx = mem->idxg[i][j];
            int is = mem->idxs_g[i][j];
            VEL(mem->maskgl+i, j) = VEL(nlp_in->dmask+i, idx) != 0.0 ? 1.0 : 0.0;
            VEL(mem->maskgu+i, j) = VEL(nlp_in->dmask+i, ni0+idx) != 0.0 ? 1.0 : 0.0;
            VEL(mem->gl+i, j) = VEL(mem->maskgl+i, j) != 0.0 ? VEL(&model->d, idx) : -INFINITY;
            VEL(mem->gu+i, j) = VEL(mem->maskgu+i, j) != 0.0 ? VEL(&model->d, ni0+idx) : INFINITY;
            // the slacks of soft rows relax the bounded sides, sig >= ls from the slack bounds of d
            VEL(mem->masksl+i, j) = (is >= 0 && VEL(mem->maskgl+i, j) != 0.0) ? 1.0 : 0.0;
            VEL(mem->masksu+i, j) = (is >= 0 && VEL(mem->maskgu+i, j) != 0.0) ? 1.0 : 0.0;
            VEL(mem->lsl+i, j) = is >= 0 ? VEL(&model->d, 2*ni0+is) : 0.0;
            VEL(mem->lsu+i, j) = is >= 0 ? VEL(&model->d, 2*ni0+cdims->ns+is) : 0.0;
        }
    }
    return ACADOS_SUCCESS;
}



// bounded and soft sides of inequality row j of stage i: 1 lower bound, 2 upper bound, 4 soft lower, 8 soft upper
static int filterddp_row_sides(ocp_nlp_filterddp_memory *mem, int i, int j)
{
    return (VEL(mem->maskgl+i, j) != 0.0) + 2*(VEL(mem->maskgu+i, j) != 0.0)
            + 4*(VEL(mem->masksl+i, j) != 0.0) + 8*(VEL(mem->masksu+i, j) != 0.0);
}



// keep the classification of the last solve, whose update rules the warm start shifts, before the next
// solve classifies the rows again with its bounds
static void filterddp_save_classification(ocp_nlp_dims *dims, ocp_nlp_filterddp_memory *mem)
{
    for (int i = 0; i < dims->N; i++)
    {
        mem->nh_prev[i] = mem->nh[i];
        mem->ng_prev[i] = mem->ng[i];
        for (int j = 0; j < mem->nh[i]; j++)
            mem->idxh_prev[i][j] = mem->idxh[i][j];
        for (int j = 0; j < mem->ng[i]; j++)
        {
            mem->idxg_prev[i][j] = mem->idxg[i][j];
            mem->idxs_g_prev[i][j] = mem->idxs_g[i][j];
            mem->sides_prev[i][j] = filterddp_row_sides(mem, i, j);
        }
    }
}



// identity of constraint row idx of stage i across stages: the column of a bound, the position in g or in h,
// so that a row keeps its identity when the stages have different numbers of rows, as h_0 and h often do
static int filterddp_row_key(ocp_nlp_dims *dims, ocp_nlp_in *in, int i, int idx)
{
    ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
    ocp_nlp_constraints_bgh_model *model = in->constraints[i];
    int nux = dims->nu[i] + dims->nx[i];
    if (idx < cdims->nb)
        return model->idxb[idx];
    if (idx < cdims->nb + cdims->ng)
        return nux + idx - cdims->nb;
    return nux + (1 << 20) + idx - cdims->nb - cdims->ng;
}



// 1 if stage i has the constraint rows of stage k of the previous solve, in the same order, with the same
// bounded and soft sides and slack indices, so that the update rules of stage k apply to stage i row for row
static int filterddp_rows_unchanged(ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_filterddp_memory *mem, int i, int k)
{
    ocp_nlp_constraints_bgh_dims *cdims_i = dims->constraints[i];
    ocp_nlp_constraints_bgh_dims *cdims_k = dims->constraints[k];
    if (mem->nh[i] != mem->nh_prev[k] || mem->ng[i] != mem->ng_prev[k] || cdims_i->ns != cdims_k->ns)
        return 0;
    for (int j = 0; j < mem->nh[i]; j++)
    {
        if (filterddp_row_key(dims, in, i, mem->idxh[i][j]) != filterddp_row_key(dims, in, k, mem->idxh_prev[k][j]))
            return 0;
    }
    for (int j = 0; j < mem->ng[i]; j++)
    {
        if (mem->idxs_g[i][j] != mem->idxs_g_prev[k][j] || filterddp_row_sides(mem, i, j) != mem->sides_prev[k][j]
                || filterddp_row_key(dims, in, i, mem->idxg[i][j]) != filterddp_row_key(dims, in, k, mem->idxg_prev[k][j]))
            return 0;
    }
    return 1;
}



/************************************************
 * initialization
 ************************************************/

static void filterddp_evaluate_constraints_at(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_opts *nlp_opts, ocp_nlp_memory *nlp_mem, ocp_nlp_workspace *nlp_work, ocp_nlp_out *out, int i)
{
    config->constraints[i]->memory_set(config->constraints[i], dims->constraints[i], nlp_mem->constraints[i], "ux_ptr", out->ux+i);
    config->constraints[i]->compute_fun(config->constraints[i], dims->constraints[i],
            in->constraints[i], nlp_opts->constraints[i], nlp_mem->constraints[i], nlp_work->constraints[i]);
}



// value of constraint row idx of stage i from the last compute_fun of the constraints module at ux, which
// leaves fun = [d_lo - c - sig_lo; c - d_up - sig_up] masked with the soft constraint slacks sig of ux
static double filterddp_row_value(ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_memory *nlp_mem,
        ocp_nlp_filterddp_memory *mem, struct blasfeo_dvec *ux, int i, int idx)
{
    ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
    ocp_nlp_constraints_bgh_model *model = in->constraints[i];
    ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
    int ni0 = filterddp_nrows(cdims);
    int nux = dims->nu[i] + dims->nx[i];
    int is = mem->idxs_row[i][idx];
    if (VEL(in->dmask+i, idx) != 0.0)
        return VEL(&model->d, idx) - VEL(&constr_mem->fun, idx) - (is >= 0 ? VEL(ux, nux+is) : 0.0);
    return VEL(&constr_mem->fun, ni0+idx) + VEL(&model->d, ni0+idx) + (is >= 0 ? VEL(ux, nux+cdims->ns+is) : 0.0);
}



static void filterddp_evaluate_dynamics_at(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_opts *nlp_opts, ocp_nlp_memory *nlp_mem, ocp_nlp_workspace *nlp_work, ocp_nlp_out *out, int i)
{
    config->dynamics[i]->memory_set(config->dynamics[i], dims->dynamics[i], nlp_mem->dynamics[i], "ux_ptr", out->ux+i);
    config->dynamics[i]->compute_fun(config->dynamics[i], dims->dynamics[i],
            in->dynamics[i], nlp_opts->dynamics[i], nlp_mem->dynamics[i], nlp_work->dynamics[i]);
}



static double filterddp_evaluate_cost_at(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_opts *nlp_opts, ocp_nlp_memory *nlp_mem, ocp_nlp_workspace *nlp_work, ocp_nlp_out *out, int i)
{
    config->cost[i]->memory_set(config->cost[i], dims->cost[i], nlp_mem->cost[i], "ux_ptr", out->ux+i);
    config->cost[i]->compute_fun(config->cost[i], dims->cost[i], in->cost[i],
            nlp_opts->cost[i], nlp_mem->cost[i], nlp_work->cost[i]);
    double *fun = config->cost[i]->memory_get(nlp_mem->cost[i], "fun");
    return *fun;
}



static void filterddp_restore_module_pointers(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_memory *nlp_mem,
        ocp_nlp_out *out)
{
    int N = dims->N;
    for (int i = 0; i <= N; i++)
    {
        config->cost[i]->memory_set(config->cost[i], dims->cost[i], nlp_mem->cost[i], "ux_ptr", out->ux+i);
        config->constraints[i]->memory_set(config->constraints[i], dims->constraints[i], nlp_mem->constraints[i], "ux_ptr", out->ux+i);
        if (i < N)
        {
            config->dynamics[i]->memory_set(config->dynamics[i], dims->dynamics[i], nlp_mem->dynamics[i], "ux_ptr", out->ux+i);
        }
    }
}



// slacks of inequality row j of stage i from its value g at ux, as in IPOPT: a soft side takes the part of g outside
// its bound, sig = max(ls, gl - g) (lower) or max(ls, g - gu) (upper), pushed off its own bound ls, and relaxes the
// bound by sig; the row slack s is g pushed into the interior of the relaxed bounds [gl - sig_l, gu + sig_u]
// (filterddp_interior with kappa_1, kappa_2)
static void filterddp_init_row_slack(ocp_nlp_dims *dims, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem,
        struct blasfeo_dvec *ux, int i, int j, double value)
{
    ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
    int nux = dims->nu[i] + dims->nx[i];
    int is = mem->idxs_g[i][j];
    double gl = VEL(mem->gl+i, j);
    double gu = VEL(mem->gu+i, j);
    if (VEL(mem->masksl+i, j) != 0.0)
    {
        double sig = filterddp_interior(filterddp_max(VEL(mem->lsl+i, j), gl-value), VEL(mem->lsl+i, j), 0.0, 1, 0,
                opts->kappa_1, opts->kappa_2);
        VEL(ux, nux+is) = sig;
        gl -= sig;
    }
    if (VEL(mem->masksu+i, j) != 0.0)
    {
        double sig = filterddp_interior(filterddp_max(VEL(mem->lsu+i, j), value-gu), VEL(mem->lsu+i, j), 0.0, 1, 0,
                opts->kappa_1, opts->kappa_2);
        VEL(ux, nux+cdims->ns+is) = sig;
        gu += sig;
    }
    VEL(mem->s+i, j) = filterddp_interior(value, gl, gu,
            VEL(mem->maskgl+i, j) != 0.0, VEL(mem->maskgu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
}



// lower bound on the multipliers of the last solve in the barrier parameter of the mu_based initialization, as
// warm_start_mult_bound_push of IPOPT
#define FILTERDDP_MULT_BOUND_PUSH 1e-3

// one bounded side at the distance d > 0 from its bound with the multiplier z: with mu > 0 sets z = mu/d, else adds
// d max(z, FILTERDDP_MULT_BOUND_PUSH) to sum and counts the side
static void filterddp_centre_side(double d, double *z, double mu, double *sum, int *n)
{
    if (!(d > 0.0))
        return;
    if (mu > 0.0)
    {
        *z = mu/d;
    }
    else
    {
        *sum += d*filterddp_max(*z, FILTERDDP_MULT_BOUND_PUSH);
        (*n)++;
    }
}



// filterddp_centre_side on every bounded side of the iterate: the controls, the slacks of the inequality rows at their
// bounds relaxed by the soft constraint slacks, and the soft constraint slacks
static void filterddp_centre_sides(ocp_nlp_dims *dims, ocp_nlp_out *out, ocp_nlp_filterddp_memory *mem, double mu,
        double *sum, int *n)
{
    for (int i = 0; i < dims->N; i++)
    {
        int nui = dims->nu[i];
        int nux = nui + dims->nx[i];
        int nsi = ((ocp_nlp_constraints_bgh_dims *) dims->constraints[i])->ns;
        for (int j = 0; j < nui; j++)
        {
            double u = VEL(out->ux+i, j);
            if (VEL(mem->maskul+i, j) != 0.0)
                filterddp_centre_side(u - VEL(mem->ul+i, j), &VEL(mem->zl+i, j), mu, sum, n);
            if (VEL(mem->maskuu+i, j) != 0.0)
                filterddp_centre_side(VEL(mem->uu+i, j) - u, &VEL(mem->zu+i, j), mu, sum, n);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            int is = mem->idxs_g[i][j];
            double s = VEL(mem->s+i, j);
            double sigl = VEL(mem->masksl+i, j) != 0.0 ? VEL(out->ux+i, nux+is) : 0.0;
            double sigu = VEL(mem->masksu+i, j) != 0.0 ? VEL(out->ux+i, nux+nsi+is) : 0.0;
            if (VEL(mem->maskgl+i, j) != 0.0)
                filterddp_centre_side(s - VEL(mem->gl+i, j) + sigl, &VEL(mem->zsl+i, j), mu, sum, n);
            if (VEL(mem->maskgu+i, j) != 0.0)
                filterddp_centre_side(VEL(mem->gu+i, j) - s + sigu, &VEL(mem->zsu+i, j), mu, sum, n);
            if (VEL(mem->masksl+i, j) != 0.0)
                filterddp_centre_side(sigl - VEL(mem->lsl+i, j), &VEL(mem->xil+i, j), mu, sum, n);
            if (VEL(mem->masksu+i, j) != 0.0)
                filterddp_centre_side(sigu - VEL(mem->lsu+i, j), &VEL(mem->xiu+i, j), mu, sum, n);
        }
    }
}



// every bound multiplier ineq_dual_init on a side with a bound and 0 on the others, phi and nu 0
static void filterddp_constant_multipliers(ocp_nlp_dims *dims, ocp_nlp_filterddp_opts *opts,
        ocp_nlp_filterddp_memory *mem)
{
    for (int i = 0; i < dims->N; i++)
    {
        blasfeo_dvecse(mem->nh[i], 0.0, mem->phi+i, 0);
        blasfeo_dvecse(mem->ng[i], 0.0, mem->nu+i, 0);
        for (int j = 0; j < dims->nu[i]; j++)
        {
            VEL(mem->zl+i, j) = opts->ineq_dual_init*VEL(mem->maskul+i, j);
            VEL(mem->zu+i, j) = opts->ineq_dual_init*VEL(mem->maskuu+i, j);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            VEL(mem->zsl+i, j) = opts->ineq_dual_init*VEL(mem->maskgl+i, j);
            VEL(mem->zsu+i, j) = opts->ineq_dual_init*VEL(mem->maskgu+i, j);
            VEL(mem->xil+i, j) = opts->ineq_dual_init*VEL(mem->masksl+i, j);
            VEL(mem->xiu+i, j) = opts->ineq_dual_init*VEL(mem->masksu+i, j);
        }
    }
}



/*
 * Multipliers and barrier parameter at the initial primal iterate, by bound_mult_init_method as in IPOPT:
 * constant: every bound multiplier ineq_dual_init (IPOPT's bound_mult_init_val), the multipliers phi of the equality
 * rows and nu of the inequality rows 0, mu = mu_init;
 * mu_based: centred, mu = avg_j d_j max(z_j, FILTERDDP_MULT_BOUND_PUSH) clipped to [mu_min, mu_init], over the bounded
 * sides j with d_j their distance to the bound at the initial iterate and z_j their multiplier of the last solve at
 * the same stage and index (ineq_dual_init without one, which gives mu = mu_init if the distances average at least
 * mu_init/ineq_dual_init, IPOPT's mu_based); then every bound multiplier mu/d_j, nu = zsu - zsl (stationarity with
 * respect to the slack) and phi = 0.
 */
static void filterddp_initialize_multipliers(ocp_nlp_dims *dims, ocp_nlp_out *out, ocp_nlp_filterddp_opts *opts,
        ocp_nlp_filterddp_memory *mem)
{
    if (opts->bound_mult_init_method != 1)
    {
        filterddp_constant_multipliers(dims, opts, mem);
        mem->mu = opts->mu_init;
        return;
    }

    // mu_based: the barrier parameter from the multipliers of the last solve, or from ineq_dual_init
    if (!mem->duals_valid)
        filterddp_constant_multipliers(dims, opts, mem);
    double sum = 0.0;
    int n = 0;
    filterddp_centre_sides(dims, out, mem, 0.0, &sum, &n);
    double mu = opts->mu_init;
    if (n > 0)
        mu = filterddp_max(filterddp_mu_min(opts->nlp_opts), filterddp_min(opts->mu_init, sum/n));

    // the multipliers centred at mu: zero on the sides without a bound, mu/d on the others
    filterddp_constant_multipliers(dims, opts, mem);
    filterddp_centre_sides(dims, out, mem, mu, &sum, &n);
    for (int i = 0; i < dims->N; i++)
    {
        for (int j = 0; j < mem->ng[i]; j++)
            VEL(mem->nu+i, j) = VEL(mem->zsu+i, j) - VEL(mem->zsl+i, j);
    }
    mem->mu = mu;
}



/*
 * Initial iterate of a solve from the controls in out: x_0 from the initial state bound, the controls pushed into
 * the interior of their bounds, the states rolled out, x_{i+1} = f(x_i, u_i). The slacks start from the constraint
 * values at this primal iterate (filterddp_init_row_slack), the multipliers and the barrier parameter as set by
 * bound_mult_init_method (filterddp_initialize_multipliers).
 */
static void filterddp_initialize_trajectory(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;

    // x_0 from the initial state bound
    ocp_nlp_constraints_bgh_model *model0 = in->constraints[0];
    ocp_nlp_constraints_bgh_dims *cdims0 = dims->constraints[0];
    for (int j = 0; j < cdims0->nbx; j++)
    {
        int col = model0->idxb[cdims0->nbu+j];
        VEL(out->ux+0, col) = VEL(&model0->d, cdims0->nbu+j);
    }

    for (int i = 0; i < N; i++)
    {
        // controls into the interior
        for (int j = 0; j < nu[i]; j++)
        {
            VEL(out->ux+i, j) = filterddp_interior(VEL(out->ux+i, j), VEL(mem->ul+i, j), VEL(mem->uu+i, j),
                    VEL(mem->maskul+i, j) != 0.0, VEL(mem->maskuu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
        }

        // slacks from constraint values
        if (mem->ng[i] > 0)
        {
            filterddp_evaluate_constraints_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, out, i);
            ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
            ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
            for (int j = 0; j < mem->ng[i]; j++)
            {
                int idx = mem->idxg[i][j];
                double value = idx < cdims->nb ? VEL(out->ux+i, constr_mem->idxb[idx])
                        : filterddp_row_value(dims, in, nlp_mem, mem, out->ux+i, i, idx);
                filterddp_init_row_slack(dims, opts, mem, out->ux+i, i, j, value);
            }
        }

        // rollout
        filterddp_evaluate_dynamics_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, out, i);
        struct blasfeo_dvec *fun = config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
        // fun = f(x,u) - x_next
        blasfeo_daxpy(nx[i+1], 1.0, fun, 0, out->ux+i+1, nu[i+1], out->ux+i+1, nu[i+1]);
    }
    filterddp_restore_module_pointers(config, dims, nlp_mem, out);
    filterddp_initialize_multipliers(dims, out, opts, mem);
}



/************************************************
 * backward pass
 ************************************************/

static void filterddp_backward_pass(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    ocp_qp_in *qp_in = nlp_mem->qp_in;
    ocp_nlp_timings *nlp_timings = nlp_mem->nlp_timings;
    acados_timer timer;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;

    double reg = 0.0;
    const double mu = mem->mu;
    const double reg_1 = opts->reg_1;
    const double reg_min = opts->reg_min;
    const double reg_max = opts->reg_max;

    int unconstrained = 1;
    int nh_total = 0;
    for (int i = 0; i < N; i++)
    {
        if (mem->nh[i] > 0 || mem->ng[i] > 0)
            unconstrained = 0;
        nh_total += mem->nh[i] + mem->ng[i];
    }

    while (reg <= reg_max)
    {
        mem->status_internal = FILTERDDP_STATUS_OK;

        // multipliers for the constraint and dynamics Hessians, then linearize at the nominal trajectory
        for (int i = 0; i < N; i++)
        {
            filterddp_set_lam(dims, out, mem, i, mem->phi+i, mem->nu+i);
        }
        blasfeo_dvecse(2*dims->ni[N], 0.0, out->lam+N, 0);

        // terminal value function from the terminal cost
        acados_tic(&timer);
        // dynamics multiplier of stage N-1 is Vx_N (unconstrained) or lambda_N, both equal the terminal gradient
        config->cost[N]->memory_set(config->cost[N], dims->cost[N], nlp_mem->cost[N], "ux_ptr", out->ux+N);
        config->cost[N]->update_qp_matrices(config->cost[N], dims->cost[N], in->cost[N],
                nlp_opts->cost[N], nlp_mem->cost[N], nlp_work->cost[N]);
        struct blasfeo_dvec *grad_N = config->cost[N]->memory_get(nlp_mem->cost[N], "grad");
        for (int j = 0; j < nx[N]; j++)
            VEL(&work->Vx, j) = VEL(grad_N, nu[N]+j);
        blasfeo_dgecp(nx[N], nx[N], qp_in->RSQrq+N, nu[N], nu[N], &work->Vxx, 0, 0);
        for (int j = 0; j < nx[N]; j++)
            for (int k = j+1; k < nx[N]; k++)
                EL(&work->Vxx, j, k) = EL(&work->Vxx, k, j);
        blasfeo_dveccp(nx[N], &work->Vx, 0, &work->lambda, 0);
        blasfeo_dveccp(nx[N], &work->Vx, 0, &work->Vd, 0);
        blasfeo_dveccp(nx[N], &work->Vx, 0, mem->costate+N, 0);
        double *fun_N = config->cost[N]->memory_get(nlp_mem->cost[N], "fun");

        mem->barrier_lagrangian_curr = 0.0;
        mem->primal_1_curr = 0.0;
        mem->eq_inf = 0.0;
        mem->ineq_inf = 0.0;
        mem->cs_inf_mu = 0.0;
        mem->cs_inf_0 = 0.0;
        mem->objective = *fun_N;
        double dual_inf_costate = 0.0;
        double dual_inf_value = 0.0;
        mem->expected_change_L = 0.0;
        double phi_norm = 0.0;
        double z_norm = 0.0;
        int ni_bounds = 0;

        // the dimensions may change between stages (multi-phase OCPs): Vx, Vxx of stage i+1 are of size nx[i+1],
        // fx is nx[i+1] x nx[i]. A stage without controls (the transition stage of a multi-phase OCP) has no
        // minimization, its value function is the Q function, Vxx = Cs and Vx = Qx; products over the empty
        // control dimension are spelled out, as BLASFEO does not guarantee them for an inner dimension 0.
        for (int i = N-1; i >= 0; i--)
        {
            int nxi = nx[i];
            int nui = nu[i];
            int nxp = nxi+1;
            int nhi = mem->nh[i];
            int ngi = mem->ng[i];
            int nz = nui - nhi;
            struct blasfeo_dmat *alpha_beta = mem->alpha_beta+i;
            struct blasfeo_dvec *ul = mem->ul+i;
            struct blasfeo_dvec *uu = mem->uu+i;
            struct blasfeo_dvec *gl = mem->gl+i;
            struct blasfeo_dvec *gu_ = mem->gu+i;
            struct blasfeo_dvec *s = mem->s+i;
            struct blasfeo_dvec *phi = mem->phi+i;
            struct blasfeo_dvec *nu_ = mem->nu+i;
            struct blasfeo_dvec *zl = mem->zl+i;
            struct blasfeo_dvec *zu = mem->zu+i;
            struct blasfeo_dvec *zsl = mem->zsl+i;
            struct blasfeo_dvec *zsu = mem->zsu+i;
            struct blasfeo_dvec *xil = mem->xil+i;
            struct blasfeo_dvec *xiu = mem->xiu+i;
            struct blasfeo_dvec *masksl = mem->masksl+i;
            struct blasfeo_dvec *masksu = mem->masksu+i;
            int nux = nui+nxi;
            int nsi = ((ocp_nlp_constraints_bgh_dims *) dims->constraints[i])->ns;
            struct blasfeo_dmat *Hs = ngi > 0 ? &work->Hsolve : &work->H;
            struct blasfeo_dmat *Rs = ngi > 0 ? &work->rhs_u_solve : &work->rhs_u;

            for (int j = 0; j < nui; j++)
                ni_bounds += (VEL(mem->maskul+i, j) != 0.0) + (VEL(mem->maskuu+i, j) != 0.0);
            for (int j = 0; j < ngi; j++)
                ni_bounds += (VEL(mem->maskgl+i, j) != 0.0) + (VEL(mem->maskgu+i, j) != 0.0)
                        + (VEL(masksl, j) != 0.0) + (VEL(masksu, j) != 0.0);

            // linearize stage i with the current dynamics multiplier: the costate lambda_{i+1} or the value
            // gradient Vx_{i+1}, equal at a solution
            {
                struct blasfeo_dvec *pi_i;
                int mode = opts->dynamics_multiplier;
                if (mode == 0)
                    pi_i = unconstrained ? &work->Vx : &work->lambda;
                else if (mode == 1)
                    pi_i = &work->lambda;
                else if (mode == 2)
                    pi_i = &work->Vx;
                else if (mode == 3)
                    pi_i = filterddp_norm_inf(nx[i+1], &work->Vx, 0) <= filterddp_norm_inf(nx[i+1], &work->lambda, 0)
                            ? &work->Vx : &work->lambda;
                else
                {
                    for (int j = 0; j < nx[i+1]; j++)
                    {
                        double v = VEL(&work->Vx, j);
                        double l = VEL(&work->lambda, j);
                        VEL(&work->pi_tmp, j) = fabs(v) <= fabs(l) ? v : l;
                    }
                    pi_i = &work->pi_tmp;
                }
                filterddp_set_pi(dims, out, i, pi_i);
            }
            config->dynamics[i]->update_qp_matrices(config->dynamics[i], dims->dynamics[i],
                    in->dynamics[i], nlp_opts->dynamics[i], nlp_mem->dynamics[i], nlp_work->dynamics[i]);
            config->cost[i]->update_qp_matrices(config->cost[i], dims->cost[i], in->cost[i],
                    nlp_opts->cost[i], nlp_mem->cost[i], nlp_work->cost[i]);
            config->constraints[i]->update_qp_matrices(config->constraints[i], dims->constraints[i],
                    in->constraints[i], nlp_opts->constraints[i], nlp_mem->constraints[i], nlp_work->constraints[i]);

            struct blasfeo_dvec *cost_grad = config->cost[i]->memory_get(nlp_mem->cost[i], "grad");
            double *cost_fun = config->cost[i]->memory_get(nlp_mem->cost[i], "fun");
            mem->objective += *cost_fun;

            for (int j = 0; j < nxi; j++)
                VEL(&work->lx, j) = VEL(cost_grad, nui+j);
            for (int j = 0; j < nui; j++)
                VEL(&work->lu, j) = VEL(cost_grad, j);

            // BAbt = [fu'; fx'; b']
            blasfeo_dgetr(nui, nx[i+1], qp_in->BAbt+i, 0, 0, &work->fu, 0, 0);
            blasfeo_dgetr(nxi, nx[i+1], qp_in->BAbt+i, nui, 0, &work->fx, 0, 0);

            filterddp_gather_constraints(config, dims, in, nlp_opts, nlp_mem, nlp_work, mem, i, &work->h, &work->g);
            filterddp_gather_jacobians(config, dims, nlp_mem, mem, i, work);

            if (nhi > 0)
            {
                mem->primal_1_curr += filterddp_norm_1(nhi, &work->h, 0);
                mem->eq_inf = filterddp_max(mem->eq_inf, filterddp_norm_inf(nhi, &work->h, 0));
            }
            if (ngi > 0)
            {
                for (int j = 0; j < ngi; j++)
                    VEL(&work->q, j) = VEL(&work->g, j) - VEL(s, j);
                mem->primal_1_curr += filterddp_norm_1(ngi, &work->q, 0);
                mem->ineq_inf = filterddp_max(mem->ineq_inf, filterddp_norm_inf(ngi, &work->q, 0));
            }

            // barrier Lagrangian and complementarity errors
            {
                double log_l = 0.0;
                double log_u = 0.0;
                for (int j = 0; j < nui; j++)
                {
                    VEL(&work->ul_dist, j) = VEL(out->ux+i, j) - VEL(ul, j);
                    VEL(&work->uu_dist, j) = VEL(uu, j) - VEL(out->ux+i, j);
                    if (VEL(mem->maskul+i, j) != 0.0)
                    {
                        log_l += log(VEL(&work->ul_dist, j));
                        double cs = VEL(&work->ul_dist, j)*VEL(zl, j);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        VEL(&work->inv_ul, j) = 1.0/VEL(&work->ul_dist, j);
                    }
                    else
                    {
                        VEL(&work->inv_ul, j) = 0.0;
                    }
                    if (VEL(mem->maskuu+i, j) != 0.0)
                    {
                        log_u += log(VEL(&work->uu_dist, j));
                        double cs = VEL(&work->uu_dist, j)*VEL(zu, j);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        VEL(&work->inv_uu, j) = 1.0/VEL(&work->uu_dist, j);
                    }
                    else
                    {
                        VEL(&work->inv_uu, j) = 0.0;
                    }
                }
                mem->barrier_lagrangian_curr -= mu*(log_l + log_u);
            }
            if (ngi > 0)
            {
                double log_l = 0.0;
                double log_u = 0.0;
                for (int j = 0; j < ngi; j++)
                {
                    // the soft constraint slacks relax the bounds of s
                    int is = mem->idxs_g[i][j];
                    double sigl = VEL(masksl, j) != 0.0 ? VEL(out->ux+i, nux+is) : 0.0;
                    double sigu = VEL(masksu, j) != 0.0 ? VEL(out->ux+i, nux+nsi+is) : 0.0;
                    VEL(&work->sl_dist, j) = VEL(s, j) - VEL(gl, j) + sigl;
                    VEL(&work->su_dist, j) = VEL(gu_, j) - VEL(s, j) + sigu;
                    if (VEL(mem->maskgl+i, j) != 0.0)
                    {
                        log_l += log(VEL(&work->sl_dist, j));
                        double cs = VEL(&work->sl_dist, j)*VEL(zsl, j);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        VEL(&work->inv_sl, j) = 1.0/VEL(&work->sl_dist, j);
                    }
                    else
                    {
                        VEL(&work->inv_sl, j) = 0.0;
                    }
                    if (VEL(mem->maskgu+i, j) != 0.0)
                    {
                        log_u += log(VEL(&work->su_dist, j));
                        double cs = VEL(&work->su_dist, j)*VEL(zsu, j);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        VEL(&work->inv_su, j) = 1.0/VEL(&work->su_dist, j);
                    }
                    else
                    {
                        VEL(&work->inv_su, j) = 0.0;
                    }
                }
                mem->barrier_lagrangian_curr -= mu*(log_l + log_u);
            }

            // soft rows: barrier terms of the slack bounds sig >= ls and elimination of the slacks. With
            // el = sigl - lsl, Xil = xil/el, SigmasL = zsl/dl and Dl = Zl + SigmasL + Xil the Newton step of the
            // slack is d sigl = -(Qsigl + SigmasL ds)/Dl, so SigmasL (Zl + Xil)/Dl replaces SigmasL in the
            // elimination of s and Qs gains -SigmasL Qsigl/Dl. The upper side is the same with s - gu in place of
            // gl - s. The slack cost is part of the stage cost through the slacks in ux.
            if (ngi > 0)
            {
                double log_l = 0.0;
                double log_u = 0.0;
                for (int j = 0; j < ngi; j++)
                {
                    int is = mem->idxs_g[i][j];
                    VEL(&work->inv_el, j) = 0.0;
                    VEL(&work->inv_eu, j) = 0.0;
                    VEL(&work->Xil, j) = 0.0;
                    VEL(&work->Xiu, j) = 0.0;
                    VEL(&work->Qsigl, j) = 0.0;
                    VEL(&work->Qsigu, j) = 0.0;
                    VEL(&work->rsigl, j) = 0.0;
                    VEL(&work->rsigu, j) = 0.0;
                    VEL(&work->inv_Dl, j) = 0.0;
                    VEL(&work->inv_Du, j) = 0.0;
                    VEL(&work->ksigl, j) = 0.0;
                    VEL(&work->ksigu, j) = 0.0;
                    VEL(&work->csigl, j) = 0.0;
                    VEL(&work->csigu, j) = 0.0;
                    VEL(&work->wsl, j) = 1.0;
                    VEL(&work->wsu, j) = 1.0;
                    if (VEL(masksl, j) != 0.0)
                    {
                        double el = VEL(out->ux+i, nux+is) - VEL(mem->lsl+i, j);
                        double cs = el*VEL(xil, j);
                        log_l += log(el);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        double inv_el = 1.0/el;
                        double Xi = VEL(xil, j)*inv_el;
                        double Sigma = VEL(&work->inv_sl, j)*VEL(zsl, j);
                        double Z = VEL(qp_in->Z+i, is);
                        double grad = VEL(cost_grad, nux+is);
                        double inv_D = 1.0/(Z + Sigma + Xi);
                        VEL(&work->inv_el, j) = inv_el;
                        VEL(&work->Xil, j) = Xi;
                        VEL(&work->Qsigl, j) = grad - mu*VEL(&work->inv_sl, j) - mu*inv_el;
                        VEL(&work->rsigl, j) = grad - VEL(zsl, j) - VEL(xil, j);
                        VEL(&work->inv_Dl, j) = inv_D;
                        VEL(&work->ksigl, j) = Sigma*inv_D;
                        VEL(&work->csigl, j) = Sigma*inv_D*VEL(&work->Qsigl, j);
                        VEL(&work->wsl, j) = (Z + Xi)*inv_D;
                    }
                    if (VEL(masksu, j) != 0.0)
                    {
                        double eu = VEL(out->ux+i, nux+nsi+is) - VEL(mem->lsu+i, j);
                        double cs = eu*VEL(xiu, j);
                        log_u += log(eu);
                        mem->cs_inf_0 = filterddp_max(mem->cs_inf_0, fabs(cs));
                        mem->cs_inf_mu = filterddp_max(mem->cs_inf_mu, fabs(cs-mu));
                        double inv_eu = 1.0/eu;
                        double Xi = VEL(xiu, j)*inv_eu;
                        double Sigma = VEL(&work->inv_su, j)*VEL(zsu, j);
                        double Z = VEL(qp_in->Z+i, nsi+is);
                        double grad = VEL(cost_grad, nux+nsi+is);
                        double inv_D = 1.0/(Z + Sigma + Xi);
                        VEL(&work->inv_eu, j) = inv_eu;
                        VEL(&work->Xiu, j) = Xi;
                        VEL(&work->Qsigu, j) = grad - mu*VEL(&work->inv_su, j) - mu*inv_eu;
                        VEL(&work->rsigu, j) = grad - VEL(zsu, j) - VEL(xiu, j);
                        VEL(&work->inv_Du, j) = inv_D;
                        VEL(&work->ksigu, j) = Sigma*inv_D;
                        VEL(&work->csigu, j) = Sigma*inv_D*VEL(&work->Qsigu, j);
                        VEL(&work->wsu, j) = (Z + Xi)*inv_D;
                    }
                }
                mem->barrier_lagrangian_curr -= mu*(log_l + log_u);
            }

            // Qu = lu + fu' Vx + mu (inv_uu - inv_ul)
            blasfeo_dgemv_t(nx[i+1], nui, 1.0, &work->fu, 0, 0, &work->Vx, 0, 1.0, &work->lu, 0, &work->Qu, 0);
            for (int j = 0; j < nui; j++)
                VEL(&work->Qu, j) += mu*(VEL(&work->inv_uu, j) - VEL(&work->inv_ul, j));
            // Qs = -nu + mu (inv_su - inv_sl), with the soft constraint slacks eliminated
            for (int j = 0; j < ngi; j++)
                VEL(&work->Qs, j) = -VEL(nu_, j) + mu*(VEL(&work->inv_su, j) - VEL(&work->inv_sl, j))
                        - VEL(&work->csigl, j) + VEL(&work->csigu, j);

            // Lagrangian Hessian blocks from the lower triangle of RSQrq, u first
            blasfeo_dgecp(nxi, nxi, qp_in->RSQrq+i, nui, nui, &work->C, 0, 0);
            blasfeo_dgecp(nui, nui, qp_in->RSQrq+i, 0, 0, &work->H, 0, 0);
            blasfeo_dgetr(nxi, nui, qp_in->RSQrq+i, nui, 0, &work->B, 0, 0);
            for (int j = 0; j < nxi; j++)
                for (int k = j+1; k < nxi; k++)
                    EL(&work->C, j, k) = EL(&work->C, k, j);
            for (int j = 0; j < nui; j++)
                for (int k = j+1; k < nui; k++)
                    EL(&work->H, j, k) = EL(&work->H, k, j);

            // C += fx' Vxx fx
            blasfeo_dgemm_tn(nxi, nxi, nx[i+1], 1.0, &work->fx, 0, 0, &work->Vxx, 0, 0, 0.0, &work->xx_tmp, 0, 0, &work->xx_tmp, 0, 0);
            blasfeo_dgemm_nn(nxi, nxi, nx[i+1], 1.0, &work->xx_tmp, 0, 0, &work->fx, 0, 0, 1.0, &work->C, 0, 0, &work->C, 0, 0);
            // ux_tmp = fu' Vxx
            blasfeo_dgemm_tn(nui, nxi, nx[i+1], 1.0, &work->fu, 0, 0, &work->Vxx, 0, 0, 0.0, &work->ux_tmp, 0, 0, &work->ux_tmp, 0, 0);

            for (int j = 0; j < nui; j++)
            {
                VEL(&work->SigmaL, j) = VEL(&work->inv_ul, j)*VEL(zl, j);
                VEL(&work->SigmaU, j) = VEL(&work->inv_uu, j)*VEL(zu, j);
            }
            for (int j = 0; j < ngi; j++)
            {
                VEL(&work->SigmasL, j) = VEL(&work->inv_sl, j)*VEL(zsl, j)*VEL(&work->wsl, j);
                VEL(&work->SigmasU, j) = VEL(&work->inv_su, j)*VEL(zsu, j)*VEL(&work->wsu, j);
                VEL(&work->Sigmas, j) = VEL(&work->SigmasL, j) + VEL(&work->SigmasU, j);
            }

            // H += Sigma_L + Sigma_U + fu' Vxx fu
            blasfeo_ddiaad(nui, 1.0, &work->SigmaL, 0, &work->H, 0, 0);
            blasfeo_ddiaad(nui, 1.0, &work->SigmaU, 0, &work->H, 0, 0);
            blasfeo_dgemm_nn(nui, nui, nx[i+1], 1.0, &work->ux_tmp, 0, 0, &work->fu, 0, 0, 1.0, &work->H, 0, 0, &work->H, 0, 0);
            // B += fu' Vxx fx
            blasfeo_dgemm_nn(nui, nxi, nx[i+1], 1.0, &work->ux_tmp, 0, 0, &work->fx, 0, 0, 1.0, &work->B, 0, 0, &work->B, 0, 0);

            if (nhi > 0)
            {
                mem->barrier_lagrangian_curr += blasfeo_ddot(nhi, &work->h, 0, phi, 0);
                blasfeo_dgemv_t(nhi, nui, 1.0, &work->hu, 0, 0, phi, 0, 1.0, &work->Qu, 0, &work->Qu, 0);
            }
            if (ngi > 0)
            {
                mem->barrier_lagrangian_curr += blasfeo_ddot(ngi, &work->q, 0, nu_, 0);
                blasfeo_dgemv_t(ngi, nui, 1.0, &work->gu, 0, 0, nu_, 0, 1.0, &work->Qu, 0, &work->Qu, 0);
            }

            if (reg != 0.0)
                blasfeo_ddiare(nui, reg, &work->H, 0, 0);

            // rhs_u = [-Qu -B]
            for (int j = 0; j < nui; j++)
            {
                EL(&work->rhs_u, j, 0) = -VEL(&work->Qu, j);
                for (int k = 0; k < nxi; k++)
                    EL(&work->rhs_u, j, k+1) = -EL(&work->B, j, k);
            }

            if (ngi > 0)
            {
                for (int j = 0; j < ngi; j++)
                {
                    EL(&work->rhs_s, j, 0) = -VEL(&work->Qs, j);
                    EL(&work->rhs_g, j, 0) = -VEL(&work->q, j);
                    for (int k = 0; k < nxi; k++)
                    {
                        EL(&work->rhs_s, j, k+1) = 0.0;
                        EL(&work->rhs_g, j, k+1) = -EL(&work->gx, j, k);
                    }
                }
                blasfeo_dgetr(ngi, nui, &work->gu, 0, 0, &work->guT, 0, 0);
                blasfeo_dgemm_nd(nui, ngi, 1.0, &work->guT, 0, 0, &work->Sigmas, 0, 0.0, &work->guT_S, 0, 0, &work->guT_S, 0, 0);
                blasfeo_dgemm_nn(nui, nui, ngi, 1.0, &work->guT_S, 0, 0, &work->gu, 0, 0, 1.0, &work->H, 0, 0, &work->Hsolve, 0, 0);
                for (int j = 0; j < ngi; j++)
                    for (int k = 0; k < nxp; k++)
                        EL(&work->rhs_sg, j, k) = EL(&work->rhs_s, j, k) + VEL(&work->Sigmas, j)*EL(&work->rhs_g, j, k);
                blasfeo_dgemm_nn(nui, nxp, ngi, 1.0, &work->guT, 0, 0, &work->rhs_sg, 0, 0, 1.0, &work->rhs_u, 0, 0, &work->rhs_u_solve, 0, 0);
            }

            filterddp_symmetrize_from_upper(nui, Hs);

            int factor_ok;
            if (nhi > 0)
            {
                blasfeo_dgelqf(nhi, nui, &work->hu, 0, 0, &work->lq, 0, 0, work->lq_work);
                blasfeo_dorglq(nui, nui, nhi, &work->lq, 0, 0, &work->Q, 0, 0, work->orglq_work);
                blasfeo_dgetr(nhi, nui, &work->Q, 0, 0, &work->Y, 0, 0);
                if (nz > 0)
                    blasfeo_dgetr(nz, nui, &work->Q, nhi, 0, &work->Z, 0, 0);
                for (int j = 0; j < nhi; j++)
                {
                    EL(&work->aby_tmp, j, 0) = -VEL(&work->h, j);
                    for (int k = 0; k < nxi; k++)
                        EL(&work->aby_tmp, j, k+1) = -EL(&work->hx, j, k);
                }
                blasfeo_dgemm_nn(nhi, nhi, nui, 1.0, &work->hu, 0, 0, &work->Y, 0, 0, 0.0, &work->AY, 0, 0, &work->AY, 0, 0);
                blasfeo_dgetrf_rp(nhi, nhi, &work->AY, 0, 0, &work->AY_lu, 0, 0, work->ipiv);
                blasfeo_drowpe(nhi, work->ipiv, &work->aby_tmp);
                blasfeo_dtrsm_llnu(nhi, nxp, 1.0, &work->AY_lu, 0, 0, &work->aby_tmp, 0, 0, &work->aby, 0, 0);
                blasfeo_dtrsm_lunn(nhi, nxp, 1.0, &work->AY_lu, 0, 0, &work->aby, 0, 0, &work->aby_tmp, 0, 0);
                blasfeo_dgecp(nhi, nxp, &work->aby_tmp, 0, 0, &work->aby, 0, 0);
                if (nz > 0)
                {
                    blasfeo_dgemm_tn(nz, nui, nui, 1.0, &work->Z, 0, 0, Hs, 0, 0, 0.0, &work->ZH, 0, 0, &work->ZH, 0, 0);
                    blasfeo_dgemm_nn(nz, nz, nui, 1.0, &work->ZH, 0, 0, &work->Z, 0, 0, 0.0, &work->M, 0, 0, &work->M, 0, 0);
                    filterddp_symmetrize_from_upper(nz, &work->M);
                    blasfeo_dpotrf_l(nz, &work->M, 0, 0, &work->LM, 0, 0);
                    factor_ok = filterddp_cholesky_ok(nz, &work->LM);
                }
                else
                {
                    factor_ok = 1;
                }
            }
            else
            {
                blasfeo_dpotrf_l(nui, Hs, 0, 0, &work->Lchol, 0, 0);
                factor_ok = filterddp_cholesky_ok(nui, &work->Lchol);
            }

            if (!factor_ok)
            {
                mem->status_internal = FILTERDDP_STATUS_BACKWARD_PASS_FAILED;
                if (reg == 0.0)
                    reg = mem->reg_last == 0.0 ? reg_1 : filterddp_max(reg_min, opts->kappa_w_m*mem->reg_last);
                else
                    reg = mem->reg_last == 0.0 ? opts->kappa_bar_w_p*reg : opts->kappa_w_p*reg;
                break;
            }

            if (nhi > 0)
            {
                blasfeo_dgemm_nn(nui, nhi, nui, 1.0, Hs, 0, 0, &work->Y, 0, 0, 0.0, &work->HY, 0, 0, &work->HY, 0, 0);
                blasfeo_dgemm_nn(nui, nxp, nhi, -1.0, &work->HY, 0, 0, &work->aby, 0, 0, 1.0, Rs, 0, 0, &work->tmp_u, 0, 0);
                blasfeo_dgemm_nn(nui, nxp, nhi, 1.0, &work->Y, 0, 0, &work->aby, 0, 0, 0.0, alpha_beta, 0, 0, alpha_beta, 0, 0);
                if (nz > 0)
                {
                    blasfeo_dgemm_tn(nz, nxp, nui, 1.0, &work->Z, 0, 0, &work->tmp_u, 0, 0, 0.0, &work->abz_tmp, 0, 0, &work->abz_tmp, 0, 0);
                    blasfeo_dtrsm_llnn(nz, nxp, 1.0, &work->LM, 0, 0, &work->abz_tmp, 0, 0, &work->abz, 0, 0);
                    blasfeo_dtrsm_lltn(nz, nxp, 1.0, &work->LM, 0, 0, &work->abz, 0, 0, &work->abz_tmp, 0, 0);
                    blasfeo_dgemm_nn(nui, nxp, nz, 1.0, &work->Z, 0, 0, &work->abz_tmp, 0, 0, 1.0, alpha_beta, 0, 0, alpha_beta, 0, 0);
                }
                blasfeo_dgemm_nn(nui, nxp, nui, -1.0, Hs, 0, 0, alpha_beta, 0, 0, 1.0, Rs, 0, 0, &work->tmp_u, 0, 0);
                blasfeo_dgemm_tn(nhi, nxp, nui, 1.0, &work->Y, 0, 0, &work->tmp_u, 0, 0, 0.0, &work->yr, 0, 0, &work->yr, 0, 0);
                blasfeo_dtrsm_lutn(nhi, nxp, 1.0, &work->AY_lu, 0, 0, &work->yr, 0, 0, &work->yr_tmp, 0, 0);
                blasfeo_dtrsm_lltu(nhi, nxp, 1.0, &work->AY_lu, 0, 0, &work->yr_tmp, 0, 0, &work->yr, 0, 0);
                blasfeo_drowpei(nhi, work->ipiv, &work->yr);
                blasfeo_dgecp(nhi, nxp, &work->yr, 0, 0, mem->psih_omegah+i, 0, 0);
            }
            else
            {
                blasfeo_dtrsm_llnn(nui, nxp, 1.0, &work->Lchol, 0, 0, Rs, 0, 0, &work->sol_tmp, 0, 0);
                blasfeo_dtrsm_lltn(nui, nxp, 1.0, &work->Lchol, 0, 0, &work->sol_tmp, 0, 0, alpha_beta, 0, 0);
            }

            if (ngi > 0)
            {
                // ds = gu du + gx dx + q, which at a stage without controls is the linearization of the rows alone
                if (nui > 0)
                    blasfeo_dgemm_nn(ngi, nxp, nui, 1.0, &work->gu, 0, 0, alpha_beta, 0, 0, -1.0, &work->rhs_g, 0, 0, mem->alphas_betas+i, 0, 0);
                else
                    blasfeo_dgecpsc(ngi, nxp, -1.0, &work->rhs_g, 0, 0, mem->alphas_betas+i, 0, 0);
                for (int j = 0; j < ngi; j++)
                    for (int k = 0; k < nxp; k++)
                        EL(mem->psig_omegag+i, j, k) = VEL(&work->Sigmas, j)*EL(mem->alphas_betas+i, j, k) - EL(&work->rhs_s, j, k);
            }

            for (int j = 0; j < nui; j++)
            {
                for (int k = 0; k < nxp; k++)
                {
                    EL(mem->chil_zetal+i, j, k) = -EL(alpha_beta, j, k)*VEL(&work->SigmaL, j);
                    EL(mem->chiu_zetau+i, j, k) = EL(alpha_beta, j, k)*VEL(&work->SigmaU, j);
                }
                EL(mem->chil_zetal+i, j, 0) = VEL(&work->inv_ul, j)*mu - VEL(zl, j) - VEL(&work->SigmaL, j)*EL(alpha_beta, j, 0);
                EL(mem->chiu_zetau+i, j, 0) = VEL(&work->inv_uu, j)*mu - VEL(zu, j) + VEL(&work->SigmaU, j)*EL(alpha_beta, j, 0);
            }
            for (int j = 0; j < ngi; j++)
            {
                struct blasfeo_dmat *ab_s = mem->alphas_betas+i;
                for (int k = 0; k < nxp; k++)
                {
                    EL(mem->chisl_zetasl+i, j, k) = -EL(ab_s, j, k)*VEL(&work->SigmasL, j);
                    EL(mem->chisu_zetasu+i, j, k) = EL(ab_s, j, k)*VEL(&work->SigmasU, j);
                }
                EL(mem->chisl_zetasl+i, j, 0) = VEL(&work->inv_sl, j)*mu - VEL(zsl, j) + VEL(&work->csigl, j) - VEL(&work->SigmasL, j)*EL(ab_s, j, 0);
                EL(mem->chisu_zetasu+i, j, 0) = VEL(&work->inv_su, j)*mu - VEL(zsu, j) + VEL(&work->csigu, j) + VEL(&work->SigmasU, j)*EL(ab_s, j, 0);
                // soft constraint slacks and their bound multipliers: d sig = -(Qsig + Sigma ds)/D, d xi = mu/e - xi - Xi d sig
                if (VEL(masksl, j) != 0.0)
                {
                    for (int k = 0; k < nxp; k++)
                    {
                        EL(mem->sigl_rule+i, j, k) = -VEL(&work->ksigl, j)*EL(ab_s, j, k);
                        EL(mem->xil_rule+i, j, k) = -VEL(&work->Xil, j)*EL(mem->sigl_rule+i, j, k);
                    }
                    EL(mem->sigl_rule+i, j, 0) -= VEL(&work->Qsigl, j)*VEL(&work->inv_Dl, j);
                    EL(mem->xil_rule+i, j, 0) = mu*VEL(&work->inv_el, j) - VEL(xil, j) - VEL(&work->Xil, j)*EL(mem->sigl_rule+i, j, 0);
                }
                if (VEL(masksu, j) != 0.0)
                {
                    for (int k = 0; k < nxp; k++)
                    {
                        EL(mem->sigu_rule+i, j, k) = VEL(&work->ksigu, j)*EL(ab_s, j, k);
                        EL(mem->xiu_rule+i, j, k) = -VEL(&work->Xiu, j)*EL(mem->sigu_rule+i, j, k);
                    }
                    EL(mem->sigu_rule+i, j, 0) -= VEL(&work->Qsigu, j)*VEL(&work->inv_Du, j);
                    EL(mem->xiu_rule+i, j, 0) = mu*VEL(&work->inv_eu, j) - VEL(xiu, j) - VEL(&work->Xiu, j)*EL(mem->sigu_rule+i, j, 0);
                }
            }

            // stationarity residuals with the costate (Lu_costate) and with the primal-dual value gradient
            // (Lu) as dynamics multiplier
            for (int j = 0; j < nui; j++)
                VEL(&work->Lu, j) = VEL(&work->lu, j) - VEL(zl, j) + VEL(zu, j);
            if (nhi > 0)
                blasfeo_dgemv_t(nhi, nui, 1.0, &work->hu, 0, 0, phi, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
            if (ngi > 0)
            {
                blasfeo_dgemv_t(ngi, nui, 1.0, &work->gu, 0, 0, nu_, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
                for (int j = 0; j < ngi; j++)
                    VEL(&work->Ls, j) = -VEL(nu_, j) - VEL(zsl, j) + VEL(zsu, j);
                double Ls_norm = filterddp_max(filterddp_norm_inf(ngi, &work->Ls, 0),
                        filterddp_max(filterddp_norm_inf(ngi, &work->rsigl, 0), filterddp_norm_inf(ngi, &work->rsigu, 0)));
                dual_inf_costate = filterddp_max(dual_inf_costate, Ls_norm);
                dual_inf_value = filterddp_max(dual_inf_value, Ls_norm);
                // the slack residuals enter the adjoint residual of the value gradient through the slack feedback
                for (int j = 0; j < ngi; j++)
                    VEL(&work->Ls, j) += -VEL(&work->ksigl, j)*VEL(&work->rsigl, j) + VEL(&work->ksigu, j)*VEL(&work->rsigu, j);
            }
            blasfeo_dgemv_t(nx[i+1], nui, 1.0, &work->fu, 0, 0, &work->lambda, 0, 1.0, &work->Lu, 0, &work->Lu_costate, 0);
            blasfeo_dgemv_t(nx[i+1], nui, 1.0, &work->fu, 0, 0, &work->Vd, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
            dual_inf_costate = filterddp_max(dual_inf_costate, filterddp_norm_inf(nui, &work->Lu_costate, 0));
            dual_inf_value = filterddp_max(dual_inf_value, filterddp_norm_inf(nui, &work->Lu, 0));
            for (int j = 0; j < nui; j++)
                z_norm += VEL(zl, j) + VEL(zu, j);
            for (int j = 0; j < ngi; j++)
                z_norm += VEL(zsl, j) + VEL(zsu, j) + VEL(xil, j) + VEL(xiu, j);
            for (int j = 0; j < nhi; j++)
                phi_norm += fabs(VEL(phi, j));
            for (int j = 0; j < ngi; j++)
                phi_norm += fabs(VEL(nu_, j));

            // value function recursion. With the reduced blocks Hs = H + gu' Sigmas gu, Bs = B + gu' Sigmas gx,
            // Cs = C + gx' Sigmas gx of the stage KKT system [Hs A; A' 0] [beta; omega] = -[Bs; cx], the paper's
            // P = Cs + beta' Hs beta + Bs' beta + beta' Bs reduces to the Schur complement
            // P = Cs + beta' Bs + omega' cx, symmetric only up to the residual of the KKT solve.
            if (opts->symmetric_value_hessian == 2)
            {
                // the same Schur complement through the factorization: with beta = betaY + Z aZ, betaY = Y aY,
                // P = Cs + (betaY' Bs + Bs' betaY) + betaY' Hs betaY - W' W, W = LM^{-1} Z' (Bs + Hs betaY);
                // without equality rows P = Cs - W' W, W = L^{-1} Bs. Rs = [-Qu_s, -Bs], sol_tmp = L^{-1} Rs and
                // abz = LM^{-1} Z' (Rs - Hs betaY) hold -W in their columns 1:nx.
                if (ngi > 0)
                {
                    blasfeo_dgemm_dn(ngi, nxi, 1.0, &work->Sigmas, 0, &work->gx, 0, 0, 0.0, &work->sg_tmp, 0, 0, &work->sg_tmp, 0, 0);
                    blasfeo_dgemm_tn(nxi, nxi, ngi, 1.0, &work->gx, 0, 0, &work->sg_tmp, 0, 0, 1.0, &work->C, 0, 0, &work->C, 0, 0);
                }
                if (nhi > 0)
                {
                    blasfeo_dgemm_nn(nui, nxi, nhi, 1.0, &work->Y, 0, 0, &work->aby, 0, 1, 0.0, &work->betaY, 0, 0, &work->betaY, 0, 0);
                    blasfeo_dgemm_nn(nui, nxi, nhi, 1.0, &work->HY, 0, 0, &work->aby, 0, 1, 0.0, &work->HbetaY, 0, 0, &work->HbetaY, 0, 0);
                    // xx_tmp = betaY' Bs = -betaY' Rs[:, 1:]
                    blasfeo_dgemm_tn(nxi, nxi, nui, -1.0, &work->betaY, 0, 0, Rs, 0, 1, 0.0, &work->xx_tmp, 0, 0, &work->xx_tmp, 0, 0);
                    for (int j = 0; j < nxi; j++)
                        for (int k = 0; k < nxi; k++)
                            EL(&work->Vxx, j, k) = EL(&work->C, j, k) + EL(&work->xx_tmp, j, k) + EL(&work->xx_tmp, k, j);
                    blasfeo_dgemm_tn(nxi, nxi, nui, 1.0, &work->betaY, 0, 0, &work->HbetaY, 0, 0, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);
                    if (nz > 0)
                        blasfeo_dgemm_tn(nxi, nxi, nz, -1.0, &work->abz, 0, 1, &work->abz, 0, 1, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);
                }
                else if (nui > 0)
                {
                    blasfeo_dgemm_tn(nxi, nxi, nui, -1.0, &work->sol_tmp, 0, 1, &work->sol_tmp, 0, 1, 1.0, &work->C, 0, 0, &work->Vxx, 0, 0);
                }
                else
                {
                    blasfeo_dgecp(nxi, nxi, &work->C, 0, 0, &work->Vxx, 0, 0);
                }
                // exactly symmetric: the rounding asymmetry of the products is mirrored away
                for (int j = 0; j < nxi; j++)
                    for (int k = j+1; k < nxi; k++)
                        EL(&work->Vxx, j, k) = EL(&work->Vxx, k, j);
            }
            else
            {
                if (nui > 0)
                    blasfeo_dgemm_tn(nxi, nxi, nui, 1.0, alpha_beta, 0, 1, &work->B, 0, 0, 1.0, &work->C, 0, 0, &work->Vxx, 0, 0);
                else
                    blasfeo_dgecp(nxi, nxi, &work->C, 0, 0, &work->Vxx, 0, 0);
                if (nhi > 0)
                    blasfeo_dgemm_tn(nxi, nxi, nhi, 1.0, mem->psih_omegah+i, 0, 1, &work->hx, 0, 0, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);
                if (ngi > 0)
                    blasfeo_dgemm_tn(nxi, nxi, ngi, 1.0, mem->psig_omegag+i, 0, 1, &work->gx, 0, 0, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);
            }
            if (opts->symmetric_value_hessian == 1)
            {
                for (int j = 0; j < nxi; j++)
                {
                    for (int k = j+1; k < nxi; k++)
                    {
                        double v = 0.5*(EL(&work->Vxx, j, k) + EL(&work->Vxx, k, j));
                        EL(&work->Vxx, j, k) = v;
                        EL(&work->Vxx, k, j) = v;
                    }
                }
            }

            if (nui > 0)
                blasfeo_dgemv_t(nui, nxi, 1.0, alpha_beta, 0, 1, &work->Qu, 0, 1.0, &work->lx, 0, &work->Vx_next, 0);
            else
                blasfeo_dveccp(nxi, &work->lx, 0, &work->Vx_next, 0);
            blasfeo_dgemv_t(nx[i+1], nxi, 1.0, &work->fx, 0, 0, &work->Vx, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
            blasfeo_dgemv_t(nx[i+1], nxi, 1.0, &work->fx, 0, 0, &work->lambda, 0, 1.0, &work->lx, 0, &work->lambda_next, 0);
            if (nhi > 0)
            {
                blasfeo_dgemv_t(nhi, nxi, 1.0, &work->hx, 0, 0, phi, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
                blasfeo_dgemv_t(nhi, nxi, 1.0, mem->psih_omegah+i, 0, 1, &work->h, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
                blasfeo_dgemv_t(nhi, nxi, 1.0, &work->hx, 0, 0, phi, 0, 1.0, &work->lambda_next, 0, &work->lambda_next, 0);
            }
            if (ngi > 0)
            {
                blasfeo_dgemv_t(ngi, nxi, 1.0, mem->alphas_betas+i, 0, 1, &work->Qs, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
                blasfeo_dgemv_t(ngi, nxi, 1.0, &work->gx, 0, 0, nu_, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
                blasfeo_dgemv_t(ngi, nxi, 1.0, mem->psig_omegag+i, 0, 1, &work->q, 0, 1.0, &work->Vx_next, 0, &work->Vx_next, 0);
                blasfeo_dgemv_t(ngi, nxi, 1.0, &work->gx, 0, 0, nu_, 0, 1.0, &work->lambda_next, 0, &work->lambda_next, 0);
            }
            // primal-dual value gradient: the recursion of Vx with the stationarity residuals Lu, Ls
            // (bound duals) in place of the barrier gradients Qu, Qs, which differ by (s z - mu)/s at
            // active bounds. As a multiplier its adjoint equation residual is
            // beta' Lu + betas' Ls + omegah' h + omegag' q, bounded by the other residuals.
            // pi holds it until the export, which picks the multiplier attaining dual_inf.
            filterddp_set_pi(dims, out, i, &work->Vd);
            if (nui > 0)
                blasfeo_dgemv_t(nui, nxi, 1.0, alpha_beta, 0, 1, &work->Lu, 0, 0.0, &work->tmp_nv, 0, &work->tmp_nv, 0);
            else
                blasfeo_dvecse(nxi, 0.0, &work->tmp_nv, 0);
            if (nhi > 0)
                blasfeo_dgemv_t(nhi, nxi, 1.0, mem->psih_omegah+i, 0, 1, &work->h, 0, 1.0, &work->tmp_nv, 0, &work->tmp_nv, 0);
            if (ngi > 0)
            {
                blasfeo_dgemv_t(ngi, nxi, 1.0, mem->alphas_betas+i, 0, 1, &work->Ls, 0, 1.0, &work->tmp_nv, 0, &work->tmp_nv, 0);
                blasfeo_dgemv_t(ngi, nxi, 1.0, mem->psig_omegag+i, 0, 1, &work->q, 0, 1.0, &work->tmp_nv, 0, &work->tmp_nv, 0);
            }
            dual_inf_value = filterddp_max(dual_inf_value, filterddp_norm_inf(nxi, &work->tmp_nv, 0));
            blasfeo_dgemv_t(nx[i+1], nxi, 1.0, &work->fx, 0, 0, &work->Vd, 0, 1.0, &work->lx, 0, &work->Vd_next, 0);
            if (nhi > 0)
                blasfeo_dgemv_t(nhi, nxi, 1.0, &work->hx, 0, 0, phi, 0, 1.0, &work->Vd_next, 0, &work->Vd_next, 0);
            if (ngi > 0)
                blasfeo_dgemv_t(ngi, nxi, 1.0, &work->gx, 0, 0, nu_, 0, 1.0, &work->Vd_next, 0, &work->Vd_next, 0);
            blasfeo_daxpy(nxi, 1.0, &work->tmp_nv, 0, &work->Vd_next, 0, &work->Vd, 0);
            blasfeo_dveccp(nxi, &work->Vx_next, 0, &work->Vx, 0);
            blasfeo_dveccp(nxi, &work->lambda_next, 0, &work->lambda, 0);
            blasfeo_dveccp(nxi, &work->lambda, 0, mem->costate+i, 0);

            for (int j = 0; j < nui; j++)
                mem->expected_change_L += VEL(&work->Qu, j)*EL(alpha_beta, j, 0);
            for (int j = 0; j < nhi; j++)
                mem->expected_change_L += VEL(&work->h, j)*EL(mem->psih_omegah+i, j, 0);
            for (int j = 0; j < ngi; j++)
                mem->expected_change_L += VEL(&work->Qs, j)*EL(mem->alphas_betas+i, j, 0) + VEL(&work->q, j)*EL(mem->psig_omegag+i, j, 0)
                        - VEL(&work->Qsigl, j)*VEL(&work->Qsigl, j)*VEL(&work->inv_Dl, j)
                        - VEL(&work->Qsigu, j)*VEL(&work->Qsigu, j)*VEL(&work->inv_Du, j);
        }
        nlp_timings->time_lin += acados_toc(&timer);

        mem->ni_bounds = ni_bounds;
        double scaling_dual = filterddp_max(opts->s_max, (phi_norm + z_norm)/filterddp_max((double) (ni_bounds + nh_total), 1.0))/opts->s_max;
        double scaling_cs = filterddp_max(opts->s_max, z_norm/filterddp_max((double) ni_bounds, 1.0))/opts->s_max;
        // both multipliers bound the stationarity error: the costate recursion carries the barrier
        // gradients of the controls, the primal-dual value gradient the gain-amplified adjoint residual.
        // Take the smaller bound.
        mem->stationarity_costate = dual_inf_costate <= dual_inf_value;
        mem->dual_inf = filterddp_min(dual_inf_costate, dual_inf_value)/scaling_dual;
        mem->primal_inf = filterddp_max(mem->eq_inf, mem->ineq_inf);
        mem->cs_inf_0 /= scaling_cs;
        mem->cs_inf_mu /= scaling_cs;
        mem->barrier_lagrangian_curr += mem->objective;
        if (mem->status_internal == FILTERDDP_STATUS_OK)
            break;
    }
    mem->reg_last = reg;
}



/************************************************
 * forward pass
 ************************************************/

static void filterddp_apply_rule(int m, int nx, struct blasfeo_dvec *ybar, int ybar_i, double gamma,
        struct blasfeo_dmat *K, struct blasfeo_dvec *xi, struct blasfeo_dvec *y, int y_i)
{
    if (m <= 0)
        return;
    for (int j = 0; j < m; j++)
        VEL(y, y_i+j) = VEL(ybar, ybar_i+j) + gamma*EL(K, j, 0);
    blasfeo_dgemv_n(m, nx, 1.0, K, 0, 1, xi, 0, 1.0, y, y_i, y, y_i);
}



static void filterddp_rollout(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_out *trial, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem,
        ocp_nlp_filterddp_workspace *work, double tau, double step_size)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    const double mu = mem->mu;
    const double one_minus_tau = 1.0 - tau;

    mem->status_internal = FILTERDDP_STATUS_OK;
    mem->primal_1_next = 0.0;
    mem->barrier_lagrangian_next = 0.0;

    blasfeo_dveccp(nx[0], out->ux+0, nu[0], trial->ux+0, nu[0]);

    for (int i = 0; i < N; i++)
    {
        int nxi = nx[i];
        int nui = nu[i];
        int nhi = mem->nh[i];
        int ngi = mem->ng[i];
        int nux = nui+nxi;
        int nsi = ((ocp_nlp_constraints_bgh_dims *) dims->constraints[i])->ns;

        for (int j = 0; j < nxi; j++)
            VEL(&work->xi, j) = VEL(trial->ux+i, nui+j) - VEL(out->ux+i, nui+j);

        blasfeo_dveccp(2*nsi, out->ux+i, nux, trial->ux+i, nux);
        filterddp_apply_rule(nui, nxi, out->ux+i, 0, step_size, mem->alpha_beta+i, &work->xi, trial->ux+i, 0);
        filterddp_apply_rule(nhi, nxi, mem->phi+i, 0, step_size, mem->psih_omegah+i, &work->xi, mem->phi_trial+i, 0);
        filterddp_apply_rule(nui, nxi, mem->zl+i, 0, step_size, mem->chil_zetal+i, &work->xi, mem->zl_trial+i, 0);
        filterddp_apply_rule(nui, nxi, mem->zu+i, 0, step_size, mem->chiu_zetau+i, &work->xi, mem->zu_trial+i, 0);
        if (ngi > 0)
        {
            filterddp_apply_rule(ngi, nxi, mem->s+i, 0, step_size, mem->alphas_betas+i, &work->xi, mem->s_trial+i, 0);
            filterddp_apply_rule(ngi, nxi, mem->nu+i, 0, step_size, mem->psig_omegag+i, &work->xi, mem->nu_trial+i, 0);
            filterddp_apply_rule(ngi, nxi, mem->zsl+i, 0, step_size, mem->chisl_zetasl+i, &work->xi, mem->zsl_trial+i, 0);
            filterddp_apply_rule(ngi, nxi, mem->zsu+i, 0, step_size, mem->chisu_zetasu+i, &work->xi, mem->zsu_trial+i, 0);
            for (int j = 0; j < ngi; j++)
            {
                int is = mem->idxs_g[i][j];
                if (VEL(mem->masksl+i, j) != 0.0)
                {
                    VEL(trial->ux+i, nux+is) = VEL(out->ux+i, nux+is) + filterddp_rule_value(nxi, mem->sigl_rule+i, j, &work->xi, step_size);
                    VEL(mem->xil_trial+i, j) = VEL(mem->xil+i, j) + filterddp_rule_value(nxi, mem->xil_rule+i, j, &work->xi, step_size);
                }
                if (VEL(mem->masksu+i, j) != 0.0)
                {
                    VEL(trial->ux+i, nux+nsi+is) = VEL(out->ux+i, nux+nsi+is) + filterddp_rule_value(nxi, mem->sigu_rule+i, j, &work->xi, step_size);
                    VEL(mem->xiu_trial+i, j) = VEL(mem->xiu+i, j) + filterddp_rule_value(nxi, mem->xiu_rule+i, j, &work->xi, step_size);
                }
            }
        }

        if (!filterddp_all_finite(nux+2*nsi, trial->ux+i, 0) || !filterddp_all_finite(ngi, mem->s_trial+i, 0))
        {
            mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
            return;
        }

        if (nhi > 0 || ngi > 0)
        {
            filterddp_evaluate_constraints_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
            for (int j = 0; j < nhi; j++)
                VEL(&work->h, j) = filterddp_row_value(dims, in, nlp_mem, mem, trial->ux+i, i, mem->idxh[i][j]);
            for (int j = 0; j < ngi; j++)
                VEL(&work->g, j) = filterddp_row_value(dims, in, nlp_mem, mem, trial->ux+i, i, mem->idxg[i][j]);
            if (nhi > 0)
            {
                if (!filterddp_all_finite(nhi, &work->h, 0))
                {
                    mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
                    return;
                }
                double n1 = 0.0;
                double d = 0.0;
                for (int j = 0; j < nhi; j++)
                {
                    n1 += fabs(VEL(&work->h, j));
                    d += VEL(&work->h, j)*VEL(mem->phi_trial+i, j);
                }
                mem->primal_1_next += n1;
                mem->barrier_lagrangian_next += d;
            }
            if (ngi > 0)
            {
                for (int j = 0; j < ngi; j++)
                    VEL(&work->q, j) = VEL(&work->g, j) - VEL(mem->s_trial+i, j);
                if (!filterddp_all_finite(ngi, &work->q, 0))
                {
                    mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
                    return;
                }
                double n1 = 0.0;
                double d = 0.0;
                for (int j = 0; j < ngi; j++)
                {
                    n1 += fabs(VEL(&work->q, j));
                    d += VEL(&work->q, j)*VEL(mem->nu_trial+i, j);
                }
                mem->primal_1_next += n1;
                mem->barrier_lagrangian_next += d;
            }
        }

        // fraction to boundary
        for (int j = 0; j < nui; j++)
        {
            double ul = VEL(trial->ux+i, j) - VEL(mem->ul+i, j);
            double uu = VEL(mem->uu+i, j) - VEL(trial->ux+i, j);
            double ul_bar = VEL(out->ux+i, j) - VEL(mem->ul+i, j);
            double uu_bar = VEL(mem->uu+i, j) - VEL(out->ux+i, j);
            if ((VEL(mem->maskul+i, j) != 0.0 && ul_bar*one_minus_tau > ul) ||
                (VEL(mem->maskuu+i, j) != 0.0 && uu_bar*one_minus_tau > uu))
            {
                mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                return;
            }
        }
        for (int j = 0; j < nui; j++)
        {
            if (VEL(mem->zl+i, j)*one_minus_tau > VEL(mem->zl_trial+i, j) ||
                VEL(mem->zu+i, j)*one_minus_tau > VEL(mem->zu_trial+i, j))
            {
                mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                return;
            }
        }
        if (ngi > 0)
        {
            for (int j = 0; j < ngi; j++)
            {
                int is = mem->idxs_g[i][j];
                int softl = VEL(mem->masksl+i, j) != 0.0;
                int softu = VEL(mem->masksu+i, j) != 0.0;
                double sl = VEL(mem->s_trial+i, j) - VEL(mem->gl+i, j) + (softl ? VEL(trial->ux+i, nux+is) : 0.0);
                double su = VEL(mem->gu+i, j) - VEL(mem->s_trial+i, j) + (softu ? VEL(trial->ux+i, nux+nsi+is) : 0.0);
                double sl_bar = VEL(mem->s+i, j) - VEL(mem->gl+i, j) + (softl ? VEL(out->ux+i, nux+is) : 0.0);
                double su_bar = VEL(mem->gu+i, j) - VEL(mem->s+i, j) + (softu ? VEL(out->ux+i, nux+nsi+is) : 0.0);
                if ((VEL(mem->maskgl+i, j) != 0.0 && sl_bar*one_minus_tau > sl) ||
                    (VEL(mem->maskgu+i, j) != 0.0 && su_bar*one_minus_tau > su))
                {
                    mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                    return;
                }
                if ((softl && (VEL(out->ux+i, nux+is) - VEL(mem->lsl+i, j))*one_minus_tau > VEL(trial->ux+i, nux+is) - VEL(mem->lsl+i, j)) ||
                    (softu && (VEL(out->ux+i, nux+nsi+is) - VEL(mem->lsu+i, j))*one_minus_tau > VEL(trial->ux+i, nux+nsi+is) - VEL(mem->lsu+i, j)))
                {
                    mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                    return;
                }
            }
            for (int j = 0; j < ngi; j++)
            {
                if (VEL(mem->zsl+i, j)*one_minus_tau > VEL(mem->zsl_trial+i, j) ||
                    VEL(mem->zsu+i, j)*one_minus_tau > VEL(mem->zsu_trial+i, j) ||
                    VEL(mem->xil+i, j)*one_minus_tau > VEL(mem->xil_trial+i, j) ||
                    VEL(mem->xiu+i, j)*one_minus_tau > VEL(mem->xiu_trial+i, j))
                {
                    mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                    return;
                }
            }
        }

        {
            double log_l = 0.0;
            double log_u = 0.0;
            for (int j = 0; j < nui; j++)
            {
                if (VEL(mem->maskul+i, j) != 0.0)
                    log_l += log(VEL(trial->ux+i, j) - VEL(mem->ul+i, j));
                if (VEL(mem->maskuu+i, j) != 0.0)
                    log_u += log(VEL(mem->uu+i, j) - VEL(trial->ux+i, j));
            }
            mem->barrier_lagrangian_next -= mu*log_l;
            mem->barrier_lagrangian_next -= mu*log_u;
        }
        if (ngi > 0)
        {
            double log_l = 0.0;
            double log_u = 0.0;
            for (int j = 0; j < ngi; j++)
            {
                int is = mem->idxs_g[i][j];
                if (VEL(mem->masksl+i, j) != 0.0)
                {
                    log_l += log(VEL(mem->s_trial+i, j) - VEL(mem->gl+i, j) + VEL(trial->ux+i, nux+is));
                    log_l += log(VEL(trial->ux+i, nux+is) - VEL(mem->lsl+i, j));
                }
                else if (VEL(mem->maskgl+i, j) != 0.0)
                    log_l += log(VEL(mem->s_trial+i, j) - VEL(mem->gl+i, j));
                if (VEL(mem->masksu+i, j) != 0.0)
                {
                    log_u += log(VEL(mem->gu+i, j) - VEL(mem->s_trial+i, j) + VEL(trial->ux+i, nux+nsi+is));
                    log_u += log(VEL(trial->ux+i, nux+nsi+is) - VEL(mem->lsu+i, j));
                }
                else if (VEL(mem->maskgu+i, j) != 0.0)
                    log_u += log(VEL(mem->gu+i, j) - VEL(mem->s_trial+i, j));
            }
            mem->barrier_lagrangian_next -= mu*log_l;
            mem->barrier_lagrangian_next -= mu*log_u;
        }

        double stage_value = filterddp_evaluate_cost_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
        if (!isfinite(stage_value))
        {
            mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
            return;
        }
        mem->barrier_lagrangian_next += stage_value;

        filterddp_evaluate_dynamics_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
        struct blasfeo_dvec *fun = config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
        // the dynamics module computes fun = f(x,u) - x_next with x_next taken from ux1, which points to out
        blasfeo_daxpy(nx[i+1], 1.0, fun, 0, out->ux+i+1, nu[i+1], trial->ux+i+1, nu[i+1]);
        if (!filterddp_all_finite(nx[i+1], trial->ux+i+1, nu[i+1]))
        {
            mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
            return;
        }
    }

    double terminal_value = filterddp_evaluate_cost_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, N);
    if (!isfinite(terminal_value))
    {
        mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
        return;
    }
    mem->barrier_lagrangian_next += terminal_value;
}



static void filterddp_forward_pass(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_out *trial, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem,
        ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_timings *nlp_timings = mem->nlp_mem->nlp_timings;
    acados_timer timer;
    acados_tic(&timer);

    mem->line_search_iter = 0;
    mem->status_internal = FILTERDDP_STATUS_OK;
    mem->step_size = 1.0;
    const double dL = mem->expected_change_L;
    const double tau = filterddp_max(opts->tau_min, 1.0 - mem->mu);
    const double theta_prev = mem->primal_1_curr;
    const double L_prev = mem->barrier_lagrangian_curr;

    while (mem->step_size >= DBL_EPSILON)
    {
        double gamma = mem->step_size;
        filterddp_rollout(config, dims, in, out, trial, opts, mem, work, tau, gamma);
        if (mem->status_internal != FILTERDDP_STATUS_OK)
        {
            mem->step_size *= 0.5;
            continue;
        }

        double theta = mem->primal_1_next;
        double L = mem->barrier_lagrangian_next;

        mem->status_internal = filterddp_filter_blocks(mem, theta, L) ? FILTERDDP_STATUS_FILTER_BLOCKED : FILTERDDP_STATUS_OK;
        if (mem->status_internal != FILTERDDP_STATUS_OK)
        {
            mem->step_size *= 0.5;
            mem->line_search_iter++;
            continue;
        }

        mem->switching = (dL < 0.0) &&
            (pow(-gamma*dL, opts->s_L)*pow(gamma, 1.0 - opts->s_L) > opts->delta*pow(theta_prev, opts->s_theta));
        mem->armijo_passed = L - L_prev - 10.0*DBL_EPSILON*fabs(L_prev) <= opts->eta_L*gamma*dL;
        if (theta <= mem->theta_min && mem->switching)
        {
            mem->status_internal = mem->armijo_passed ? FILTERDDP_STATUS_OK : FILTERDDP_STATUS_STEP_ACCEPTANCE;
        }
        else
        {
            int suff = (theta <= (1.0 - opts->gamma_theta)*theta_prev) || (L <= L_prev - opts->gamma_L*theta_prev);
            mem->status_internal = suff ? FILTERDDP_STATUS_OK : FILTERDDP_STATUS_FORWARD_FAILED;
        }
        if (mem->status_internal != FILTERDDP_STATUS_OK)
        {
            mem->step_size *= 0.5;
            mem->line_search_iter++;
            continue;
        }
        break;
    }
    if (mem->step_size < DBL_EPSILON)
        mem->status_internal = FILTERDDP_STATUS_LINE_SEARCH_FAILED;

    nlp_timings->time_glob += acados_toc(&timer);
}



static void filterddp_accept_trial(ocp_nlp_dims *dims, ocp_nlp_out *out, ocp_nlp_out *trial, ocp_nlp_filterddp_memory *mem)
{
    int N = dims->N;
    for (int i = 0; i <= N; i++)
    {
        blasfeo_dveccp(dims->nv[i], trial->ux+i, 0, out->ux+i, 0);
    }
    for (int i = 0; i < N; i++)
    {
        blasfeo_dveccp(mem->ng[i], mem->s_trial+i, 0, mem->s+i, 0);
        blasfeo_dveccp(mem->nh[i], mem->phi_trial+i, 0, mem->phi+i, 0);
        blasfeo_dveccp(mem->ng[i], mem->nu_trial+i, 0, mem->nu+i, 0);
        blasfeo_dveccp(dims->nu[i], mem->zl_trial+i, 0, mem->zl+i, 0);
        blasfeo_dveccp(dims->nu[i], mem->zu_trial+i, 0, mem->zu+i, 0);
        blasfeo_dveccp(mem->ng[i], mem->zsl_trial+i, 0, mem->zsl+i, 0);
        blasfeo_dveccp(mem->ng[i], mem->zsu_trial+i, 0, mem->zsu+i, 0);
        blasfeo_dveccp(mem->ng[i], mem->xil_trial+i, 0, mem->xil+i, 0);
        blasfeo_dveccp(mem->ng[i], mem->xiu_trial+i, 0, mem->xiu+i, 0);
    }
}



/************************************************
 * output
 ************************************************/

// stage of the previous solve whose update rules stage i takes in the warm start: the next stage if it has the
// same dimensions, else stage i itself (the last stage, and the stages before a change of nx or nu in a
// multi-phase OCP, where the rules of the next stage do not fit)
static int filterddp_shift_source(ocp_nlp_dims *dims, int i)
{
    if (i+1 < dims->N && dims->nx[i+1] == dims->nx[i] && dims->nu[i+1] == dims->nu[i])
        return i+1;
    return i;
}



/*
 * Warm start: initialize the iterate from the affine update rules of the previous solve shifted by one
 * stage, u_i = u_{i+1} + alpha_{i+1} + beta_{i+1} (x_i - x_{i+1}), rolled out from the new initial state,
 * with the same rules for slacks and multipliers. A stage without a next stage of the same dimensions keeps its
 * own rule of the previous solve, evaluated at its new state: the last stage, and in a multi-phase OCP the
 * stages before a change of nx or nu, as the transition stage and the stage before it. The constraint rows may
 * differ between stages and between solves (h_0 with fewer rows than h, phases with different constraints): each
 * row takes the rule of the row with the same identity and the same bounded and soft sides at the source stage
 * of the previous solve, and rows without one start as in a cold start, from their value at the rolled out point
 * with the initial multipliers.
 */
static int filterddp_shift_policy(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_out *trial, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem,
        ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    const double dual_floor = 1e-3;
    const double gamma = mem->policy_gamma;

    mem->warm_rows_fresh = 0;

    ocp_nlp_constraints_bgh_model *model0 = in->constraints[0];
    ocp_nlp_constraints_bgh_dims *cdims0 = dims->constraints[0];
    blasfeo_dveccp(nx[0], out->ux+0, nu[0], trial->ux+0, nu[0]);
    for (int j = 0; j < cdims0->nbx; j++)
    {
        int col = model0->idxb[cdims0->nbu+j];
        VEL(trial->ux+0, col) = VEL(&model0->d, cdims0->nbu+j);
    }

    int ok = 1;
    for (int i = 0; i < N && ok; i++)
    {
        int k = filterddp_shift_source(dims, i);
        int nxi = nx[i];
        int nui = nu[i];
        int nhi = mem->nh[i];
        int ngi = mem->ng[i];
        int nux = nui+nxi;
        int nsi = ((ocp_nlp_constraints_bgh_dims *) dims->constraints[i])->ns;
        int nsk = ((ocp_nlp_constraints_bgh_dims *) dims->constraints[k])->ns;

        for (int j = 0; j < nxi; j++)
            VEL(&work->xi, j) = VEL(trial->ux+i, nui+j) - VEL(out->ux+k, nui+j);
        if (nsi == nsk)
            blasfeo_dveccp(2*nsi, out->ux+k, nux, trial->ux+i, nux);
        else
            blasfeo_dvecse(2*nsi, 0.0, trial->ux+i, nux);
        filterddp_apply_rule(nui, nxi, out->ux+k, 0, gamma, mem->alpha_beta+k, &work->xi, trial->ux+i, 0);
        filterddp_apply_rule(nui, nxi, mem->zl+k, 0, gamma, mem->chil_zetal+k, &work->xi, mem->zl_trial+i, 0);
        filterddp_apply_rule(nui, nxi, mem->zu+k, 0, gamma, mem->chiu_zetau+k, &work->xi, mem->zu_trial+i, 0);
        for (int j = 0; j < nui; j++)
        {
            VEL(trial->ux+i, j) = filterddp_interior(VEL(trial->ux+i, j), VEL(mem->ul+i, j), VEL(mem->uu+i, j),
                    VEL(mem->maskul+i, j) != 0.0, VEL(mem->maskuu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
            VEL(mem->zl_trial+i, j) = filterddp_max(VEL(mem->zl_trial+i, j), dual_floor)*VEL(mem->maskul+i, j);
            VEL(mem->zu_trial+i, j) = filterddp_max(VEL(mem->zu_trial+i, j), dual_floor)*VEL(mem->maskuu+i, j);
        }

        if (filterddp_rows_unchanged(dims, in, mem, i, k))
        {
            // the rows of stage k of the previous solve: its rules apply as they are
            filterddp_apply_rule(nhi, nxi, mem->phi+k, 0, gamma, mem->psih_omegah+k, &work->xi, mem->phi_trial+i, 0);
            if (ngi > 0)
            {
                filterddp_apply_rule(ngi, nxi, mem->s+k, 0, gamma, mem->alphas_betas+k, &work->xi, mem->s_trial+i, 0);
                filterddp_apply_rule(ngi, nxi, mem->nu+k, 0, gamma, mem->psig_omegag+k, &work->xi, mem->nu_trial+i, 0);
                filterddp_apply_rule(ngi, nxi, mem->zsl+k, 0, gamma, mem->chisl_zetasl+k, &work->xi, mem->zsl_trial+i, 0);
                filterddp_apply_rule(ngi, nxi, mem->zsu+k, 0, gamma, mem->chisu_zetasu+k, &work->xi, mem->zsu_trial+i, 0);
                for (int j = 0; j < ngi; j++)
                {
                    int is = mem->idxs_g[i][j];
                    if (VEL(mem->masksl+i, j) != 0.0)
                    {
                        VEL(trial->ux+i, nux+is) = VEL(out->ux+k, nux+is) + filterddp_rule_value(nxi, mem->sigl_rule+k, j, &work->xi, gamma);
                        VEL(mem->xil_trial+i, j) = VEL(mem->xil+k, j) + filterddp_rule_value(nxi, mem->xil_rule+k, j, &work->xi, gamma);
                    }
                    if (VEL(mem->masksu+i, j) != 0.0)
                    {
                        VEL(trial->ux+i, nux+nsi+is) = VEL(out->ux+k, nux+nsi+is) + filterddp_rule_value(nxi, mem->sigu_rule+k, j, &work->xi, gamma);
                        VEL(mem->xiu_trial+i, j) = VEL(mem->xiu+k, j) + filterddp_rule_value(nxi, mem->xiu_rule+k, j, &work->xi, gamma);
                    }
                }
            }
        }
        else
        {
            // equality rows: the shifted rule of the same row, zero multiplier for a new row
            for (int j = 0; j < nhi; j++)
            {
                int key = filterddp_row_key(dims, in, i, mem->idxh[i][j]);
                VEL(mem->phi_trial+i, j) = 0.0;
                for (int jp = 0; jp < mem->nh_prev[k]; jp++)
                {
                    if (filterddp_row_key(dims, in, k, mem->idxh_prev[k][jp]) == key)
                    {
                        VEL(mem->phi_trial+i, j) = VEL(mem->phi+k, jp)
                                + filterddp_rule_value(nxi, mem->psih_omegah+k, jp, &work->xi, gamma);
                        break;
                    }
                }
            }

            // inequality rows with the same identity and the same bounded and soft sides take the shifted rule,
            // the others are marked with a NaN slack and initialized below
            int nfresh = 0;
            for (int j = 0; j < ngi; j++)
            {
                int key = filterddp_row_key(dims, in, i, mem->idxg[i][j]);
                int sides = filterddp_row_sides(mem, i, j);
                int jp = 0;
                for (; jp < mem->ng_prev[k]; jp++)
                {
                    if (mem->sides_prev[k][jp] == sides && filterddp_row_key(dims, in, k, mem->idxg_prev[k][jp]) == key)
                        break;
                }
                if (jp == mem->ng_prev[k])
                {
                    VEL(mem->s_trial+i, j) = NAN;
                    nfresh++;
                    continue;
                }
                VEL(mem->s_trial+i, j) = VEL(mem->s+k, jp) + filterddp_rule_value(nxi, mem->alphas_betas+k, jp, &work->xi, gamma);
                VEL(mem->nu_trial+i, j) = VEL(mem->nu+k, jp) + filterddp_rule_value(nxi, mem->psig_omegag+k, jp, &work->xi, gamma);
                VEL(mem->zsl_trial+i, j) = VEL(mem->zsl+k, jp) + filterddp_rule_value(nxi, mem->chisl_zetasl+k, jp, &work->xi, gamma);
                VEL(mem->zsu_trial+i, j) = VEL(mem->zsu+k, jp) + filterddp_rule_value(nxi, mem->chisu_zetasu+k, jp, &work->xi, gamma);
                int is = mem->idxs_g[i][j];
                int isp = mem->idxs_g_prev[k][jp];
                if (sides & 4)
                {
                    VEL(trial->ux+i, nux+is) = VEL(out->ux+k, nux+isp) + filterddp_rule_value(nxi, mem->sigl_rule+k, jp, &work->xi, gamma);
                    VEL(mem->xil_trial+i, j) = VEL(mem->xil+k, jp) + filterddp_rule_value(nxi, mem->xil_rule+k, jp, &work->xi, gamma);
                }
                if (sides & 8)
                {
                    VEL(trial->ux+i, nux+nsi+is) = VEL(out->ux+k, nux+nsk+isp) + filterddp_rule_value(nxi, mem->sigu_rule+k, jp, &work->xi, gamma);
                    VEL(mem->xiu_trial+i, j) = VEL(mem->xiu+k, jp) + filterddp_rule_value(nxi, mem->xiu_rule+k, jp, &work->xi, gamma);
                }
            }
            if (nfresh > 0)
            {
                filterddp_evaluate_constraints_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
                ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
                ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
                for (int j = 0; j < ngi; j++)
                {
                    if (!isnan(VEL(mem->s_trial+i, j)))
                        continue;
                    int idx = mem->idxg[i][j];
                    int is = mem->idxs_g[i][j];
                    double value = idx < cdims->nb ? VEL(trial->ux+i, constr_mem->idxb[idx])
                            : filterddp_row_value(dims, in, nlp_mem, mem, trial->ux+i, i, idx);
                    if (VEL(mem->masksl+i, j) != 0.0)
                        VEL(trial->ux+i, nux+is) = filterddp_max(VEL(mem->lsl+i, j), VEL(mem->gl+i, j)-value);
                    if (VEL(mem->masksu+i, j) != 0.0)
                        VEL(trial->ux+i, nux+nsi+is) = filterddp_max(VEL(mem->lsu+i, j), value-VEL(mem->gu+i, j));
                    VEL(mem->s_trial+i, j) = value;
                    VEL(mem->nu_trial+i, j) = 0.0;
                    VEL(mem->zsl_trial+i, j) = opts->ineq_dual_init*VEL(mem->maskgl+i, j);
                    VEL(mem->zsu_trial+i, j) = opts->ineq_dual_init*VEL(mem->maskgu+i, j);
                    VEL(mem->xil_trial+i, j) = opts->ineq_dual_init*VEL(mem->masksl+i, j);
                    VEL(mem->xiu_trial+i, j) = opts->ineq_dual_init*VEL(mem->masksu+i, j);
                }
                mem->warm_rows_fresh += nfresh;
            }
        }
        for (int j = 0; j < ngi; j++)
        {
            int is = mem->idxs_g[i][j];
            double gl = VEL(mem->gl+i, j);
            double gu = VEL(mem->gu+i, j);
            if (VEL(mem->masksl+i, j) != 0.0)
            {
                VEL(trial->ux+i, nux+is) = filterddp_interior(VEL(trial->ux+i, nux+is), VEL(mem->lsl+i, j), 0.0, 1, 0,
                        opts->kappa_1, opts->kappa_2);
                VEL(mem->xil_trial+i, j) = filterddp_max(VEL(mem->xil_trial+i, j), dual_floor);
                gl -= VEL(trial->ux+i, nux+is);
            }
            if (VEL(mem->masksu+i, j) != 0.0)
            {
                VEL(trial->ux+i, nux+nsi+is) = filterddp_interior(VEL(trial->ux+i, nux+nsi+is), VEL(mem->lsu+i, j), 0.0, 1, 0,
                        opts->kappa_1, opts->kappa_2);
                VEL(mem->xiu_trial+i, j) = filterddp_max(VEL(mem->xiu_trial+i, j), dual_floor);
                gu += VEL(trial->ux+i, nux+nsi+is);
            }
            VEL(mem->s_trial+i, j) = filterddp_interior(VEL(mem->s_trial+i, j), gl, gu,
                    VEL(mem->maskgl+i, j) != 0.0, VEL(mem->maskgu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
            VEL(mem->zsl_trial+i, j) = filterddp_max(VEL(mem->zsl_trial+i, j), dual_floor)*VEL(mem->maskgl+i, j);
            VEL(mem->zsu_trial+i, j) = filterddp_max(VEL(mem->zsu_trial+i, j), dual_floor)*VEL(mem->maskgu+i, j);
        }
        if (!filterddp_all_finite(nux+2*nsi, trial->ux+i, 0) || !filterddp_all_finite(ngi, mem->s_trial+i, 0))
        {
            ok = 0;
            break;
        }
        filterddp_evaluate_dynamics_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
        struct blasfeo_dvec *fun = config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
        blasfeo_daxpy(nx[i+1], 1.0, fun, 0, out->ux+i+1, nu[i+1], trial->ux+i+1, nu[i+1]);
        ok = filterddp_all_finite(nx[i+1], trial->ux+i+1, nu[i+1]);
    }
    if (ok)
        filterddp_accept_trial(dims, out, trial, mem);
    filterddp_restore_module_pointers(config, dims, nlp_mem, out);
    return ok;
}



static void print_iteration(int iter, ocp_nlp_filterddp_memory *mem)
{
    if (iter % 10 == 0)
    {
        printf("  iter     objective        pr_inf       du_inf       cs_inf     lg(mu)   lg(reg)    alpha     ls\n");
    }
    if (mem->reg_last == 0.0)
    {
        printf(" %5d   %.8e   %.4e   %.4e   %.4e   % 1.2f      -     %.4e  %2d\n",
                iter, mem->objective, mem->primal_inf, mem->dual_inf, mem->cs_inf_0,
                log10(mem->mu), mem->step_size, mem->line_search_iter);
    }
    else
    {
        printf(" %5d   %.8e   %.4e   %.4e   %.4e   % 1.2f  % 2.4f   %.4e  %2d\n",
                iter, mem->objective, mem->primal_inf, mem->dual_inf, mem->cs_inf_0,
                log10(mem->mu), log10(mem->reg_last), mem->step_size, mem->line_search_iter);
    }
}



/************************************************
 * functions
 ************************************************/

static void filterddp_export_solution(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_out *out,
        ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    int N = dims->N;

    // multipliers in the acados layout: equality rows get phi split by sign over the two sides,
    // bounds on u and slacked inequality rows the interior point duals of their bounds (nu = zsu - zsl
    // at a solution), the slack bounds of soft rows their multipliers, and the initial state rows the
    // value function gradient at stage 0. The soft constraint slacks are already in ux. The dynamics
    // multipliers pi and the value function gradient are those attaining dual_inf: the costate, or the
    // primal-dual value gradient that the backward pass leaves in pi.
    struct blasfeo_dvec *value_gradient_0 = mem->stationarity_costate ? mem->costate : &work->Vd;
    for (int i = 0; i < N; i++)
    {
        if (mem->stationarity_costate)
            filterddp_set_pi(dims, out, i, mem->costate+i+1);
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        ocp_nlp_constraints_bgh_model *model = in->constraints[i];
        int ni0 = filterddp_nrows(cdims);
        filterddp_set_lam(dims, out, mem, i, mem->phi+i, mem->nu+i);
        for (int j = 0; j < cdims->nbu; j++)
        {
            if (mem->idxs_row[i][j] >= 0)
                continue;
            int col = model->idxb[j];
            VEL(out->lam+i, j) = VEL(mem->zl+i, col);
            VEL(out->lam+i, ni0+j) = VEL(mem->zu+i, col);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            int idx = mem->idxg[i][j];
            int is = mem->idxs_g[i][j];
            VEL(out->lam+i, idx) = VEL(mem->zsl+i, j);
            VEL(out->lam+i, ni0+idx) = VEL(mem->zsu+i, j);
            if (VEL(mem->masksl+i, j) != 0.0)
                VEL(out->lam+i, 2*ni0+is) = VEL(mem->xil+i, j);
            if (VEL(mem->masksu+i, j) != 0.0)
                VEL(out->lam+i, 2*ni0+cdims->ns+is) = VEL(mem->xiu+i, j);
        }
        if (i == 0)
        {
            for (int j = cdims->nbu; j < cdims->nbu+cdims->nbx; j++)
            {
                double v = VEL(value_gradient_0, model->idxb[j]-dims->nu[0]);
                VEL(out->lam+i, j) = v > 0.0 ? v : 0.0;
                VEL(out->lam+i, ni0+j) = v < 0.0 ? -v : 0.0;
            }
        }
    }

    ocp_nlp_approximate_qp_matrices(config, dims, in, out, nlp_opts, nlp_mem, nlp_work);
    ocp_nlp_approximate_qp_vectors_sqp(config, dims, in, out, nlp_opts, nlp_mem, nlp_work);
    ocp_nlp_res_compute(dims, nlp_opts, in, out, nlp_mem->nlp_res, nlp_mem, nlp_work);
    ocp_nlp_res_get_inf_norm(nlp_mem->nlp_res, &out->inf_norm_res);
    nlp_mem->cost_value = mem->objective;
}



int ocp_nlp_filterddp(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_)
{
    acados_timer timer0;
    acados_tic(&timer0);

    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_in *nlp_in = nlp_in_;
    ocp_nlp_out *nlp_out = nlp_out_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_timings *nlp_timings = nlp_mem->nlp_timings;

    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    ocp_nlp_out *trial = nlp_work->tmp_nlp_out;

    ocp_nlp_timings_reset(nlp_timings);

#if defined(ACADOS_WITH_OPENMP)
    int num_threads_bkp = omp_get_num_threads();
    omp_set_num_threads(opts->nlp_opts->num_threads);
#endif

    ocp_nlp_initialize_submodules(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);

    if (mem->policy_valid)
        filterddp_save_classification(dims, mem);
    if (filterddp_setup_bounds(dims, nlp_in, mem) != ACADOS_SUCCESS)
    {
        nlp_mem->status = ACADOS_QP_FAILURE;
        nlp_timings->time_tot = acados_toc(&timer0);
        return nlp_mem->status;
    }

    double mu_previous = mem->mu;
    int warm = opts->warm_start && mem->policy_valid
            && filterddp_shift_policy(config, dims, nlp_in, nlp_out, trial, opts, mem, work);
    mem->warm_started = warm;
    if (!warm)
    {
        // initial iterate, multipliers and barrier parameter
        mem->warm_rows_fresh = 0;
        filterddp_initialize_trajectory(config, dims, nlp_in, nlp_out, opts, mem, work);
    }
    else
    {
        // continue from the barrier parameter the previous solve ended with
        mem->mu = filterddp_max(filterddp_mu_min(nlp_opts), filterddp_min(opts->mu_init, mu_previous));
    }
    mem->reg_last = 0.0;
    mem->step_size = 0.0;
    mem->barrier_iter = 0;
    mem->line_search_iter = 0;
    mem->theta_max = opts->theta_max_factor;
    mem->theta_min = 0.0;
    mem->switching = 0;
    mem->armijo_passed = 0;
    filterddp_reset_filter(mem);
    nlp_mem->status = ACADOS_SUCCESS;

    // timeout: the clock runs from the start of the call, initialization and warm-start shift included
    if (opts->timeout_heuristic != MAX_OVERALL)
        mem->timeout_estimated_per_iteration_time = 0;
    double timeout_previous_time_tot = 0.;
    int timeout_checked = 0;

    int iter = 0;
    for (; iter < nlp_opts->max_iter; )
    {
        filterddp_backward_pass(config, dims, nlp_in, nlp_out, opts, mem, work);

        if (iter == 0 && mem->barrier_iter == 0)
        {
            mem->theta_max = opts->theta_max_factor*filterddp_max(1.0, mem->primal_1_curr);
            mem->theta_min = opts->theta_min_factor*filterddp_max(1.0, mem->primal_1_curr);
            filterddp_reset_filter(mem);
        }

        if (iter < mem->stat_m)
        {
            mem->stat[mem->stat_n*iter+0] = mem->dual_inf;
            mem->stat[mem->stat_n*iter+1] = mem->primal_inf;
            mem->stat[mem->stat_n*iter+2] = mem->cs_inf_0;
            mem->stat[mem->stat_n*iter+3] = mem->objective;
            mem->stat[mem->stat_n*iter+4] = mem->mu;
            mem->stat[mem->stat_n*iter+5] = mem->reg_last;
            mem->stat[mem->stat_n*iter+6] = mem->step_size;
            mem->stat[mem->stat_n*iter+7] = mem->line_search_iter;
        }

        if (mem->status_internal != FILTERDDP_STATUS_OK)
        {
            if (nlp_opts->print_level > 0)
                printf("Backward pass failure, unable to find an iteration matrix with correct inertia.\n");
            nlp_mem->status = ACADOS_QP_FAILURE;
            break;
        }

        if (!isfinite(mem->dual_inf) || !isfinite(mem->primal_inf) || !isfinite(mem->cs_inf_0) || !isfinite(mem->objective))
        {
            nlp_mem->status = ACADOS_NAN_DETECTED;
            break;
        }

        if (mem->dual_inf < nlp_opts->tol_stat && mem->eq_inf < nlp_opts->tol_eq
                && mem->ineq_inf < nlp_opts->tol_ineq && mem->cs_inf_0 < nlp_opts->tol_comp)
        {
            nlp_mem->status = ACADOS_SUCCESS;
            break;
        }

        // timeout, checked after each backward pass, before the barrier update or forward pass that would follow:
        // the iterate is the one the last line search accepted (or the initial one) and the update rules,
        // multipliers and statistics are those of the backward pass at it, as on convergence. An iteration in
        // the sense of the heuristics is the time between two checks, a forward pass or barrier update and the
        // backward pass after it; neither pass is interrupted.
        if (opts->timeout_max_time > 0.)
        {
            nlp_timings->time_tot = acados_toc(&timer0);

            // update the estimate of the time per iteration based on the chosen heuristic
            if (timeout_checked)
            {
                double timeout_time_prev_iter = nlp_timings->time_tot - timeout_previous_time_tot;

                switch (opts->timeout_heuristic)
                {
                    case LAST:
                        mem->timeout_estimated_per_iteration_time = timeout_time_prev_iter;
                        break;
                    case MAX_CALL:
                    case MAX_OVERALL:
                        mem->timeout_estimated_per_iteration_time = filterddp_max(timeout_time_prev_iter,
                                mem->timeout_estimated_per_iteration_time);
                        break;
                    case AVERAGE:
                        // as in SQP, the average starts from zero in each call
                        mem->timeout_estimated_per_iteration_time = 0.5*timeout_time_prev_iter
                                + 0.5*mem->timeout_estimated_per_iteration_time;
                        break;
                    case ZERO: // predicted per iteration time is zero as initialized
                        break;
                    default:
                        printf("Unknown timeout heuristic.\n");
                        exit(1);
                }
            }
            timeout_previous_time_tot = nlp_timings->time_tot;
            timeout_checked = 1;

            if (opts->timeout_max_time <= nlp_timings->time_tot + mem->timeout_estimated_per_iteration_time)
            {
                nlp_mem->status = ACADOS_TIMEOUT;
                break;
            }
        }

        // barrier subproblem solved: its error below kappa_eps*mu, where no error needs to be smaller than its
        // termination tolerance (with equal tolerances and kappa_eps >= 10 this is kappa_eps*mu, as mu > mu_min)
        double mu_min = filterddp_mu_min(nlp_opts);
        double tol_mu = opts->kappa_eps*mem->mu;
        int barrier_solved = mem->dual_inf <= filterddp_max(tol_mu, nlp_opts->tol_stat)
                && mem->eq_inf <= filterddp_max(tol_mu, nlp_opts->tol_eq)
                && mem->ineq_inf <= filterddp_max(tol_mu, nlp_opts->tol_ineq) && mem->cs_inf_mu <= tol_mu;
        if (barrier_solved && mem->ni_bounds > 0 && mem->mu > mu_min)
        {
            mem->mu = filterddp_max(mu_min, filterddp_min(opts->kappa_mu*mem->mu, pow(mem->mu, opts->theta_mu)));
            filterddp_reset_filter(mem);
            mem->barrier_iter++;
            continue;
        }

        if (nlp_opts->print_level > 0)
            print_iteration(iter, mem);

        filterddp_forward_pass(config, dims, nlp_in, nlp_out, trial, opts, mem, work);
        if (mem->status_internal != FILTERDDP_STATUS_OK)
        {
            if (nlp_opts->print_level > 0)
                printf("Line search failed to find a suitable iterate\n");
            nlp_mem->status = ACADOS_MINSTEP;
            break;
        }

        filterddp_accept_trial(dims, nlp_out, trial, mem);
        if (!mem->armijo_passed && !mem->switching)
            filterddp_update_filter(mem, opts);
        mem->barrier_lagrangian_curr = mem->barrier_lagrangian_next;
        mem->primal_1_curr = mem->primal_1_next;

        iter++;
    }

    // a solve stopped by convergence or timeout leaves the update rules of the backward pass at the returned
    // iterate, with the full feedforward still to be taken, which the warm start shifts as they are (factor 1).
    // A timeout never stops at the iteration cap, so the extra backward pass below does not add to its time.
    int policy_valid = nlp_mem->status == ACADOS_SUCCESS || nlp_mem->status == ACADOS_TIMEOUT;
    mem->policy_gamma = 1.0;
    if (iter == nlp_opts->max_iter)
    {
        nlp_mem->status = ACADOS_MAXITER;
        // the update rules belong to the iterate before the last step, which took the fraction step_size of
        // their feedforward: u = ubar + step alpha + beta (x - xbar). Either recompute them at the returned
        // iterate, or keep them: from the returned iterate, the rest of the step is (1 - step) alpha. Kept, the
        // objective and the value gradient (pi) exported below are also those of the iterate before the last
        // step. Without any step (max_iter 0) there are no rules of this solve to keep.
        if (opts->policy_at_cap || iter == 0)
        {
            filterddp_backward_pass(config, dims, nlp_in, nlp_out, opts, mem, work);
            policy_valid = mem->status_internal == FILTERDDP_STATUS_OK;
        }
        else
        {
            mem->policy_gamma = 1.0 - mem->step_size;
            policy_valid = 1;
        }
    }
    filterddp_restore_module_pointers(config, dims, nlp_mem, nlp_out);
    mem->policy_valid = policy_valid;
    // the multipliers of a solve that leaves a policy are those of the mu_based initialization of the next solve
    mem->duals_valid = policy_valid;
    filterddp_export_solution(config, dims, nlp_in, nlp_out, opts, mem, work);

    nlp_mem->iter = iter;
    if (nlp_opts->print_level > 0)
    {
        print_iteration(iter, mem);
        if (nlp_mem->status == ACADOS_SUCCESS)
            printf("\nEXIT: Optimal solution found.\n\n");
        else if (nlp_mem->status == ACADOS_MAXITER)
            printf("\nEXIT: Failed, maximum solver iterations reached.\n\n");
        else if (nlp_mem->status == ACADOS_QP_FAILURE)
            printf("\nEXIT: Failed, unable to find iteration matrix with desired inertia in backward pass.\n\n");
        else if (nlp_mem->status == ACADOS_TIMEOUT)
            printf("\nEXIT: Stopped, maximum time reached.\n\n");
        else
            printf("\nEXIT: Failed, line-search unable to find acceptable iterate in forward pass.\n\n");
    }

#if defined(ACADOS_WITH_OPENMP)
    omp_set_num_threads(num_threads_bkp);
#endif
    nlp_timings->time_tot = acados_toc(&timer0);

    return nlp_mem->status;
}



/*
 * Warm start of the next solve from the affine policy of the last solve, in closed loop from the new initial state
 * x0: stage i takes the update rule of stage k = filterddp_shift_source(i) of the last solve, the next stage (the
 * rules shifted by one stage) or its own,
 *     u_i = ubar_k + gamma alpha_k + beta_k (x_i - xbar_k),   x_{i+1} = f(x_i, u_i),
 * from x_0 = x0, with (xbar, ubar) the iterate of the last solve, gamma the feedforward still to be taken
 * (policy_gamma) and the controls pushed into the interior of their bounds as in the initialization of a solve. The
 * rollout replaces x and u of the iterate in nlp_out; the next solve initializes from these controls as from any
 * initial guess: it rolls them out from its initial state bound and sets the slacks and multipliers there. Set the
 * bounds and parameters of the next solve before this call, as the rollout uses them. Returns ACADOS_SUCCESS,
 * ACADOS_READY without a policy to take (no solve since the creation or the last reset, a failed last solve, or a
 * policy already taken), ACADOS_NAN_DETECTED if the rollout is not finite and ACADOS_QP_FAILURE if the constraints
 * cannot be classified; on failure nlp_out is unchanged.
 */
int ocp_nlp_filterddp_warm_start_from_policy(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_, double *x0)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_in *nlp_in = nlp_in_;
    ocp_nlp_out *nlp_out = nlp_out_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    ocp_nlp_out *trial = nlp_work->tmp_nlp_out;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    const double gamma = mem->policy_gamma;

    if (!mem->policy_valid)
        return ACADOS_READY;

    ocp_nlp_initialize_submodules(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);
    int status = filterddp_setup_bounds(dims, nlp_in, mem);
    if (status != ACADOS_SUCCESS)
        return status;

    blasfeo_pack_dvec(nx[0], x0, 1, trial->ux+0, nu[0]);
    int ok = filterddp_all_finite(nx[0], trial->ux+0, nu[0]);
    for (int i = 0; i < N && ok; i++)
    {
        int k = filterddp_shift_source(dims, i);
        for (int j = 0; j < nx[i]; j++)
            VEL(&work->xi, j) = VEL(trial->ux+i, nu[i]+j) - VEL(nlp_out->ux+k, nu[i]+j);
        filterddp_apply_rule(nu[i], nx[i], nlp_out->ux+k, 0, gamma, mem->alpha_beta+k, &work->xi, trial->ux+i, 0);
        for (int j = 0; j < nu[i]; j++)
        {
            VEL(trial->ux+i, j) = filterddp_interior(VEL(trial->ux+i, j), VEL(mem->ul+i, j), VEL(mem->uu+i, j),
                    VEL(mem->maskul+i, j) != 0.0, VEL(mem->maskuu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
        }
        if (!filterddp_all_finite(nu[i], trial->ux+i, 0))
        {
            ok = 0;
            break;
        }
        // the dynamics module computes fun = f(x,u) - x_next with x_next taken from nlp_out
        filterddp_evaluate_dynamics_at(config, dims, nlp_in, nlp_opts, nlp_mem, nlp_work, trial, i);
        struct blasfeo_dvec *fun = config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
        blasfeo_daxpy(nx[i+1], 1.0, fun, 0, nlp_out->ux+i+1, nu[i+1], trial->ux+i+1, nu[i+1]);
        ok = filterddp_all_finite(nx[i+1], trial->ux+i+1, nu[i+1]);
    }
    filterddp_restore_module_pointers(config, dims, nlp_mem, nlp_out);
    if (!ok)
        return ACADOS_NAN_DETECTED;

    for (int i = 0; i <= N; i++)
        blasfeo_dveccp(nu[i]+nx[i], trial->ux+i, 0, nlp_out->ux+i, 0);
    // the iterate is no longer the one of the update rules
    mem->policy_valid = 0;
    return ACADOS_SUCCESS;
}



int ocp_nlp_filterddp_setup_qp_matrices_and_factorize(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
    void *opts_, void *mem_, void *work_)
{
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_filterddp_workspace *work = work_;

    return ocp_nlp_common_setup_qp_matrices_and_factorize(config_, dims_, nlp_in_, nlp_out_, opts->nlp_opts, mem->nlp_mem, work->nlp_work);
}



void ocp_nlp_filterddp_eval_kkt_residual(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_in *nlp_in = nlp_in_;
    ocp_nlp_out *nlp_out = nlp_out_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;

    ocp_nlp_initialize_submodules(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);
    ocp_nlp_approximate_qp_matrices(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);
    ocp_nlp_approximate_qp_vectors_sqp(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);
    ocp_nlp_res_compute(dims, nlp_opts, nlp_in, nlp_out, nlp_mem->nlp_res, nlp_mem, nlp_work);
}



void ocp_nlp_filterddp_memory_reset_qp_solver(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
    void *opts_, void *mem_, void *work_)
{
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_qp_xcond_solver_config *qp_solver = config->qp_solver;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;

    // back to the state after creation: no warm start, and none of the values of earlier solves in the iterate,
    // the update rules and the workspace, which are zero as allocated. Not all of BLASFEO's products with
    // beta = 0 skip the output they overwrite (the edge kernels of dgemm_nd and dgemm_dn, all kernels of the
    // generic target), so a NaN or inf a failed solve leaves there would otherwise reach every later solve.
    int N = dims->N;
    int nx_max, nu_max, ni_max;
    filterddp_dims_max(dims, &nx_max, &nu_max, &ni_max);
    // the data of the per stage vectors and matrices and of the costate, one block in memory_assign
    char *start = (char *) mem->ul[0].mem;
    char *end = (char *) mem->costate[N].mem + mem->costate[N].memsize;
    memset(start, 0, end - start);
    // the workspace after the nlp workspace, one block in cast_workspace; the factors forget their inverse diagonal
    start = (char *) work->Vx.mem;
    end = (char *) (work->ipiv + ni_max);
    memset(start, 0, end - start);
    work->Lchol.use_dA = 0;
    work->LM.use_dA = 0;
    work->AY_lu.use_dA = 0;

    mem->mu = 0.0;
    mem->filter_size = 0;
    mem->policy_valid = 0;
    mem->duals_valid = 0;
    mem->timeout_estimated_per_iteration_time = 0;
    mem->warm_started = 0;
    mem->warm_rows_fresh = 0;
    mem->policy_gamma = 1.0;

    config->qp_solver->memory_reset(qp_solver, dims->qp_solver,
        nlp_mem->qp_in, nlp_mem->qp_out, opts->nlp_opts->qp_solver_opts,
        nlp_mem->qp_solver_mem, nlp_work->qp_work);
}



int ocp_nlp_filterddp_precompute(void *config_, void *dims_, void *nlp_in_, void *nlp_out_,
                void *opts_, void *mem_, void *work_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_in *nlp_in = nlp_in_;
    ocp_nlp_out *nlp_out = nlp_out_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;

    nlp_mem->workspace_size = ocp_nlp_workspace_calculate_size(config, dims, opts->nlp_opts, nlp_in);

    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_filterddp_cast_workspace(config, dims, opts, nlp_in, mem, work);
    ocp_nlp_workspace *nlp_work = work->nlp_work;

    int N = dims->N;
    int tmp;

    config->constraints[0]->dims_get(config->constraints[0], dims->constraints[0], "nbx", &tmp);
    if (tmp != dims->nx[0])
    {
        printf("ocp_nlp_filterddp: the initial state must be fixed through nbx_0 == nx, got nbx_0 = %d, nx = %d.\n", tmp, dims->nx[0]);
        exit(1);
    }
    if (dims->ni[N] > 0)
    {
        printf("ocp_nlp_filterddp: terminal constraints are not supported, got ni[N] = %d.\n", dims->ni[N]);
        exit(1);
    }
    for (int i = 0; i <= N; i++)
    {
        if (dims->nz[i] > 0)
        {
            printf("ocp_nlp_filterddp: algebraic variables are not supported, got nz[%d] = %d.\n", i, dims->nz[i]);
            exit(1);
        }
    }


    return ocp_nlp_precompute_common(config, dims, nlp_in, nlp_out, opts->nlp_opts, nlp_mem, nlp_work);
}



void ocp_nlp_filterddp_eval_param_sens(void *config_, void *dims_, void *opts_, void *mem_, void *work_,
                                 char *field, int stage, int index, void *sens_nlp_out_)
{
    acados_timer timer0;
    acados_tic(&timer0);

    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_out *sens_nlp_out = sens_nlp_out_;

    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;

    ocp_nlp_common_eval_param_sens(config, dims, opts->nlp_opts, nlp_mem, nlp_work,
                                 field, stage, index, sens_nlp_out);

    nlp_mem->nlp_timings->time_solution_sensitivities = acados_toc(&timer0);

    return;
}



void ocp_nlp_filterddp_eval_lagr_grad_p(void *config_, void *dims_, void *nlp_in_, void *opts_, void *mem_, void *work_,
                                 const char *field, void *lagr_grad_wrt_params)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;

    ocp_nlp_in *nlp_in = nlp_in_;

    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;

    ocp_nlp_common_eval_lagr_grad_p(config, dims, nlp_in, opts->nlp_opts, nlp_mem, nlp_work,
                                 field, lagr_grad_wrt_params);

    return;
}



void ocp_nlp_filterddp_eval_solution_sens_adj_p(void *config_, void *dims_, void *in_,
                        void *opts_, void *mem_, void *work_, void *sens_nlp_out,
                        const char *field, int stage, void *grad_p)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_in *in = in_;
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_opts *opts = opts_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_filterddp_workspace *work = work_;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    ocp_nlp_common_eval_solution_sens_adj_p(config, dims, in,
                        opts->nlp_opts, nlp_mem, nlp_work,
                        sens_nlp_out, field, stage, grad_p);
}



void ocp_nlp_filterddp_get(void *config_, void *dims_, void *mem_, const char *field, void *return_value_)
{
    ocp_nlp_config *config = config_;
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_filterddp_memory *mem = mem_;

    char *ptr_module = NULL;
    int module_length = 0;
    char module[MAX_STR_LEN];
    extract_module_name(field, module, &module_length, &ptr_module);

    if ( ptr_module!=NULL && (!strcmp(ptr_module, "time")) )
    {
        ocp_nlp_timings_get(config, mem->nlp_mem->nlp_timings, field, return_value_);
    }
    else if (!strcmp("stat", field))
    {
        double **value = return_value_;
        *value = mem->stat;
    }
    else if (!strcmp("statistics", field))
    {
        int n_row = mem->stat_m<mem->nlp_mem->iter+1 ? mem->stat_m : mem->nlp_mem->iter+1;
        double *value = return_value_;
        for (int ii=0; ii<n_row; ii++)
        {
            value[ii+0] = ii;
            for (int jj=0; jj<mem->stat_n; jj++)
                value[ii+(jj+1)*n_row] = mem->stat[jj+ii*mem->stat_n];
        }
    }
    else if (!strcmp("stat_m", field))
    {
        int *value = return_value_;
        *value = mem->stat_m;
    }
    else if (!strcmp("stat_n", field))
    {
        int *value = return_value_;
        *value = mem->stat_n;
    }
    else if (!strcmp("qp_xcond_dims", field))
    {
        void **value = return_value_;
        *value = dims->qp_solver->xcond_dims;
    }
    else if (!strcmp("filterddp_mu", field))
    {
        double *value = return_value_;
        *value = mem->mu;
    }
    else if (!strcmp("filterddp_reg_last", field))
    {
        double *value = return_value_;
        *value = mem->reg_last;
    }
    else if (!strcmp("filterddp_warm_started", field))
    {
        int *value = return_value_;
        *value = mem->warm_started;
    }
    else if (!strcmp("filterddp_warm_rows_fresh", field))
    {
        int *value = return_value_;
        *value = mem->warm_rows_fresh;
    }
    else
    {
        ocp_nlp_memory_get(config, mem->nlp_mem, field, return_value_);
    }
}



void ocp_nlp_filterddp_opts_get(void *config_, void *opts_,
                          const char *field, void *return_value_)
{
    ocp_nlp_filterddp_opts *opts = opts_;

    if (!strcmp("nlp_opts", field))
    {
        void **value = return_value_;
        *value = opts->nlp_opts;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_filterddp_opts_get\n", field);
        exit(1);
    }
}



void ocp_nlp_filterddp_work_get(void *config_, void *dims_, void *work_,
                          const char *field, void *return_value_)
{
    ocp_nlp_filterddp_workspace *work = work_;

    if (!strcmp("nlp_work", field))
    {
        void **value = return_value_;
        *value = work->nlp_work;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_filterddp_work_get\n", field);
        exit(1);
    }
}



void ocp_nlp_filterddp_terminate(void *config_, void *mem_, void *work_)
{
    ocp_nlp_config *config = config_;
    ocp_nlp_filterddp_memory *mem = mem_;
    ocp_nlp_filterddp_workspace *work = work_;

    config->qp_solver->terminate(config->qp_solver, mem->nlp_mem->qp_solver_mem, work->nlp_work->qp_work);
}



bool ocp_nlp_filterddp_is_real_time_algorithm()
{
    return false;
}



void ocp_nlp_filterddp_step_update(void *config_, void *dims_,
            void *in_, void *out_, void *qp_out_, void *opts_, void *mem_,
            void *work_, void *out_destination_, void *solver_mem,
            double alpha, bool full_step_dual)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_out *out = out_;
    ocp_nlp_out *out_destination = out_destination_;
    copy_ocp_nlp_out(dims, out, out_destination);
}



void ocp_nlp_filterddp_get_at_stage(void *config_, void *dims_, void *mem_, int stage, const char *field, void *return_value_)
{
    ocp_nlp_dims *dims = dims_;
    ocp_nlp_filterddp_memory *mem = mem_;
    double *value = return_value_;

    if (stage < 0 || stage >= dims->N)
    {
        printf("\nerror: ocp_nlp_filterddp_get_at_stage: field %s not available at stage %d\n", field, stage);
        exit(1);
    }
    int nx = dims->nx[stage];
    int nu = dims->nu[stage];

    if (!strcmp(field, "K"))
    {
        blasfeo_unpack_dmat(nu, nx, mem->alpha_beta+stage, 0, 1, value, nu);
    }
    else if (!strcmp(field, "k"))
    {
        blasfeo_unpack_dmat(nu, 1, mem->alpha_beta+stage, 0, 0, value, nu);
    }
    else
    {
        printf("\nerror: ocp_nlp_filterddp_get_at_stage: field %s not available\n", field);
        exit(1);
    }
}



void ocp_nlp_filterddp_config_initialize_default(void *config_)
{
    ocp_nlp_config *config = (ocp_nlp_config *) config_;

    config->opts_calculate_size = &ocp_nlp_filterddp_opts_calculate_size;
    config->opts_assign = &ocp_nlp_filterddp_opts_assign;
    config->opts_initialize_default = &ocp_nlp_filterddp_opts_initialize_default;
    config->opts_update = &ocp_nlp_filterddp_opts_update;
    config->opts_set = &ocp_nlp_filterddp_opts_set;
    config->opts_set_at_stage = &ocp_nlp_filterddp_opts_set_at_stage;
    config->memory_calculate_size = &ocp_nlp_filterddp_memory_calculate_size;
    config->memory_assign = &ocp_nlp_filterddp_memory_assign;
    config->workspace_calculate_size = &ocp_nlp_filterddp_workspace_calculate_size;
    config->evaluate = &ocp_nlp_filterddp;
    config->setup_qp_matrices_and_factorize = &ocp_nlp_filterddp_setup_qp_matrices_and_factorize;
    config->memory_reset_qp_solver = &ocp_nlp_filterddp_memory_reset_qp_solver;
    config->eval_param_sens = &ocp_nlp_filterddp_eval_param_sens;
    config->eval_lagr_grad_p = &ocp_nlp_filterddp_eval_lagr_grad_p;
    config->eval_solution_sens_adj_p = &ocp_nlp_filterddp_eval_solution_sens_adj_p;
    config->config_initialize_default = &ocp_nlp_filterddp_config_initialize_default;
    config->precompute = &ocp_nlp_filterddp_precompute;
    config->get = &ocp_nlp_filterddp_get;
    config->get_at_stage = &ocp_nlp_filterddp_get_at_stage;
    config->opts_get = &ocp_nlp_filterddp_opts_get;
    config->work_get = &ocp_nlp_filterddp_work_get;
    config->terminate = &ocp_nlp_filterddp_terminate;
    config->step_update = &ocp_nlp_filterddp_step_update;
    config->is_real_time_algorithm = &ocp_nlp_filterddp_is_real_time_algorithm;
    config->eval_kkt_residual = &ocp_nlp_filterddp_eval_kkt_residual;

    return;
}
