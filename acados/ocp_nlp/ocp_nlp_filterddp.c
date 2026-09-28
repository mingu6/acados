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

    opts->nlp_scaling = 0;
    opts->nlp_scaling_max_gradient = 100.0;

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
    else if (!strcmp(field, "filterddp_nlp_scaling"))
    {
        opts->nlp_scaling = *(int *) value;
    }
    else if (!strcmp(field, "filterddp_nlp_scaling_max_gradient"))
    {
        opts->nlp_scaling_max_gradient = *(double *) value;
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

    // classification
    size += 6*N*sizeof(int);
    size += 2*N*sizeof(int *);
    for (int i = 0; i < N; i++)
    {
        size += 2*ni[i]*sizeof(int);
    }

    // per stage vectors: 8 bounds, 7 iterate, 7 trial, 2 scaling
    size += 24*N*sizeof(struct blasfeo_dvec);
    // per stage matrices: 8 update rules
    size += 8*N*sizeof(struct blasfeo_dmat);
    for (int i = 0; i < N; i++)
    {
        size += 4*blasfeo_memsize_dvec(nu[i]);         // ul, uu, maskul, maskuu
        size += 4*blasfeo_memsize_dvec(ni[i]);         // gl, gu, maskgl, maskgu
        size += 2*blasfeo_memsize_dvec(ni[i]);         // s, s_trial
        size += 2*blasfeo_memsize_dvec(ni[i]);         // phi, phi_trial
        size += 2*blasfeo_memsize_dvec(ni[i]);         // nu, nu_trial
        size += 4*blasfeo_memsize_dvec(nu[i]);         // zl, zu, zl_trial, zu_trial
        size += 4*blasfeo_memsize_dvec(ni[i]);         // zsl, zsu, zsl_trial, zsu_trial
        size += 2*blasfeo_memsize_dvec(ni[i]);         // h_scale, g_scale
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
    for (int i = 0; i < N; i++)
    {
        assign_and_advance_int(ni[i], &mem->idxh[i], &c_ptr);
        assign_and_advance_int(ni[i], &mem->idxg[i], &c_ptr);
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
    assign_and_advance_blasfeo_dvec_structs(N, &mem->h_scale, &c_ptr);
    assign_and_advance_blasfeo_dvec_structs(N, &mem->g_scale, &c_ptr);

    // matrix structs
    assign_and_advance_blasfeo_dmat_structs(N, &mem->alpha_beta, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->alphas_betas, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->psih_omegah, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->psig_omegag, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chil_zetal, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chiu_zetau, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chisl_zetasl, &c_ptr);
    assign_and_advance_blasfeo_dmat_structs(N, &mem->chisu_zetasu, &c_ptr);

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
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->h_scale+i, &c_ptr);
        assign_and_advance_blasfeo_dvec_mem(ni[i], mem->g_scale+i, &c_ptr);

        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->alpha_beta+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->alphas_betas+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->psih_omegah+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->psig_omegag+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->chil_zetal+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(nu[i], nx[i]+1, mem->chiu_zetau+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->chisl_zetasl+i, &c_ptr);
        assign_and_advance_blasfeo_dmat_mem(ni[i], nx[i]+1, mem->chisu_zetasu+i, &c_ptr);
    }

    for (int i = 0; i < N; i++)
    {
        blasfeo_dvecse(ni[i], 0.0, mem->s+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->phi+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->nu+i, 0);
        blasfeo_dvecse(nu[i], 0.0, mem->zl+i, 0);
        blasfeo_dvecse(nu[i], 0.0, mem->zu+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->zsl+i, 0);
        blasfeo_dvecse(ni[i], 0.0, mem->zsu+i, 0);
        blasfeo_dvecse(ni[i], 1.0, mem->h_scale+i, 0);
        blasfeo_dvecse(ni[i], 1.0, mem->g_scale+i, 0);
    }

    mem->objective_scale = 1.0;
    mem->filter_size = 0;
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

    size += 4*blasfeo_memsize_dvec(nx_max);                 // Vx, Vx_next, lambda, lambda_next
    size += blasfeo_memsize_dmat(nx_max, nx_max);           // Vxx
    size += 2*blasfeo_memsize_dmat(nx_max, nx_max);         // fx, C
    size += blasfeo_memsize_dmat(nx_max, nu_max);           // fu
    size += 4*blasfeo_memsize_dmat(ni_max, nx_max);         // hx, gx, hxs + spare
    size += 3*blasfeo_memsize_dmat(ni_max, nu_max);         // hu, gu, hus
    size += 2*blasfeo_memsize_dvec(nx_max);                 // lx + spare
    size += 6*blasfeo_memsize_dvec(ni_max);                 // h, h_scaled, g, q, Ls, Qs
    size += 2*blasfeo_memsize_dvec(nu_max);                 // lu, Qu
    size += blasfeo_memsize_dvec(nu_max);                   // Lu
    size += 3*blasfeo_memsize_dmat(nu_max, nu_max);         // H, Hsolve, Lchol
    size += 2*blasfeo_memsize_dmat(nu_max, nx_max);         // B, ux_tmp
    size += blasfeo_memsize_dmat(nx_max, nx_max);           // xx_tmp
    size += 6*blasfeo_memsize_dvec(nu_max);                 // ul_dist ... SigmaU
    size += 7*blasfeo_memsize_dvec(ni_max);                 // sl_dist ... Sigmas
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

    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->fx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nx_max, nu_max, &work->fu, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->hx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->hu, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->gx, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->gu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nx_max, &work->lx, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->lu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->h, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->h_scaled, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->g, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->q, &c_ptr);

    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->C, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nu_max, &work->H, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->B, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nx_max, nx_max, &work->xx_tmp, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(nu_max, nx_max, &work->ux_tmp, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->Qu, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(ni_max, &work->Qs, &c_ptr);
    assign_and_advance_blasfeo_dvec_mem(nu_max, &work->Lu, &c_ptr);
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

    assign_and_advance_blasfeo_dmat_mem(ni_max, nu_max, &work->hus, &c_ptr);
    assign_and_advance_blasfeo_dmat_mem(ni_max, nx_max, &work->hxs, &c_ptr);
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
    int ni = dims->ni[i];
    blasfeo_dvecse(2*ni, 0.0, out->lam+i, 0);
    for (int j = 0; j < mem->nh[i]; j++)
    {
        double m = VEL(phi, j);
        if (m >= 0.0)
            VEL(out->lam+i, ni+mem->idxh[i][j]) = m;
        else
            VEL(out->lam+i, mem->idxh[i][j]) = -m;
    }
    for (int j = 0; j < mem->ng[i]; j++)
    {
        double m = VEL(nu_, j);
        if (m >= 0.0)
            VEL(out->lam+i, ni+mem->idxg[i][j]) = m;
        else
            VEL(out->lam+i, mem->idxg[i][j]) = -m;
    }
}



static void filterddp_set_pi(ocp_nlp_dims *dims, ocp_nlp_out *out, int i, struct blasfeo_dvec *lambda)
{
    blasfeo_dveccp(dims->nx[i+1], lambda, 0, out->pi+i, 0);
}



/*
 * Classify the constraint rows of each stage: bounds on u are handled as control limits, the initial
 * state bound fixes x_0, all other rows are equalities if both sides coincide and inequalities with a
 * slack otherwise. Bounds are user data that may change between solves, so this runs at every solve.
 */
static int filterddp_classify_constraints(ocp_nlp_dims *dims, ocp_nlp_in *nlp_in, ocp_nlp_filterddp_memory *mem)
{
    int N = dims->N;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_bgh_model *model = nlp_in->constraints[i];
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        int ni = dims->ni[i];
        int nbu = cdims->nbu;
        int nbx = cdims->nbx;
        mem->nh[i] = 0;
        mem->ng[i] = 0;
        for (int j = nbu; j < ni; j++)
        {
            if (i == 0 && j < nbu+nbx)
                continue;
            int has_lower = VEL(nlp_in->dmask+i, j) != 0.0;
            int has_upper = VEL(nlp_in->dmask+i, ni+j) != 0.0;
            double lower = VEL(&model->d, j);
            double upper = VEL(&model->d, ni+j);
            if (has_lower && has_upper && lower == upper)
            {
                mem->idxh[i][mem->nh[i]] = j;
                mem->nh[i]++;
            }
            else
            {
                mem->idxg[i][mem->ng[i]] = j;
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
            // compute_fun leaves function values in tmp_ni of the constraints workspace; recompute from ux and DCt directly
            for (int j = 0; j < mem->ng[i]; j++)
            {
                int idx = mem->idxg[i][j];
                double value;
                if (idx < cdims->nb)
                {
                    value = VEL(out->ux+i, constr_mem->idxb[idx]);
                }
                else
                {
                    // fun = d_lower - eval, so eval = d_lower - fun for unmasked lower bounds; use the upper row otherwise
                    if (VEL(in->dmask+i, idx) != 0.0)
                        value = VEL(&((ocp_nlp_constraints_bgh_model *) in->constraints[i])->d, idx) - VEL(&constr_mem->fun, idx);
                    else
                        value = VEL(&constr_mem->fun, dims->ni[i]+idx) + VEL(&((ocp_nlp_constraints_bgh_model *) in->constraints[i])->d, dims->ni[i]+idx);
                }
                VEL(mem->s+i, j) = filterddp_interior(value, VEL(mem->gl+i, j), VEL(mem->gu+i, j),
                        VEL(mem->maskgl+i, j) != 0.0, VEL(mem->maskgu+i, j) != 0.0, opts->kappa_1, opts->kappa_2);
            }
        }

        blasfeo_dvecse(mem->nh[i], 0.0, mem->phi+i, 0);
        blasfeo_dvecse(mem->ng[i], 0.0, mem->nu+i, 0);
        for (int j = 0; j < nu[i]; j++)
        {
            VEL(mem->zl+i, j) = opts->ineq_dual_init*VEL(mem->maskul+i, j);
            VEL(mem->zu+i, j) = opts->ineq_dual_init*VEL(mem->maskuu+i, j);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            VEL(mem->zsl+i, j) = opts->ineq_dual_init*VEL(mem->maskgl+i, j);
            VEL(mem->zsu+i, j) = opts->ineq_dual_init*VEL(mem->maskgu+i, j);
        }

        // rollout
        filterddp_evaluate_dynamics_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, out, i);
        struct blasfeo_dvec *fun = config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
        // fun = f(x,u) - x_next
        blasfeo_daxpy(nx[i+1], 1.0, fun, 0, out->ux+i+1, nu[i+1], out->ux+i+1, nu[i+1]);
    }
    filterddp_restore_module_pointers(config, dims, nlp_mem, out);
}



static void filterddp_compute_nlp_scaling(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
        ocp_nlp_out *out, ocp_nlp_filterddp_opts *opts, ocp_nlp_filterddp_memory *mem, ocp_nlp_filterddp_workspace *work)
{
    ocp_nlp_opts *nlp_opts = opts->nlp_opts;
    ocp_nlp_memory *nlp_mem = mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = work->nlp_work;
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    double max_gradient = opts->nlp_scaling_max_gradient;

    mem->objective_scale = 1.0;
    for (int i = 0; i < N; i++)
    {
        blasfeo_dvecse(mem->nh[i], 1.0, mem->h_scale+i, 0);
        blasfeo_dvecse(mem->ng[i], 1.0, mem->g_scale+i, 0);
    }
    if (!opts->nlp_scaling)
        return;

    ocp_nlp_approximate_qp_matrices(config, dims, in, out, nlp_opts, nlp_mem, nlp_work);

    double objective_gradient_norm = 0.0;
    for (int i = 0; i <= N; i++)
    {
        objective_gradient_norm = filterddp_max(objective_gradient_norm, filterddp_norm_inf(nu[i]+nx[i], nlp_mem->cost_grad+i, 0));
        if (i == N)
            break;
        filterddp_gather_jacobians(config, dims, nlp_mem, mem, i, work);
        for (int j = 0; j < mem->nh[i]; j++)
        {
            double gradient_norm = 0.0;
            for (int k = 0; k < nx[i]; k++)
                gradient_norm = filterddp_max(gradient_norm, fabs(EL(&work->hx, j, k)));
            for (int k = 0; k < nu[i]; k++)
                gradient_norm = filterddp_max(gradient_norm, fabs(EL(&work->hu, j, k)));
            VEL(mem->h_scale+i, j) = gradient_norm == 0.0 ? 1.0 : filterddp_min(1.0, max_gradient/gradient_norm);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            double gradient_norm = 1.0;
            for (int k = 0; k < nx[i]; k++)
                gradient_norm = filterddp_max(gradient_norm, fabs(EL(&work->gx, j, k)));
            for (int k = 0; k < nu[i]; k++)
                gradient_norm = filterddp_max(gradient_norm, fabs(EL(&work->gu, j, k)));
            VEL(mem->g_scale+i, j) = filterddp_min(1.0, max_gradient/gradient_norm);
        }
    }
    mem->objective_scale = objective_gradient_norm == 0.0 ? 1.0 : filterddp_min(1.0, max_gradient/objective_gradient_norm);
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
    const double objective_scale = mem->objective_scale;
    const double reg_1 = objective_scale*opts->reg_1;
    const double reg_min = objective_scale*opts->reg_min;
    const double reg_max = objective_scale*opts->reg_max;

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
            VEL(&work->Vx, j) = objective_scale*VEL(grad_N, nu[N]+j);
        blasfeo_dgecpsc(nx[N], nx[N], objective_scale, qp_in->RSQrq+N, nu[N], nu[N], &work->Vxx, 0, 0);
        for (int j = 0; j < nx[N]; j++)
            for (int k = j+1; k < nx[N]; k++)
                EL(&work->Vxx, j, k) = EL(&work->Vxx, k, j);
        blasfeo_dveccp(nx[N], &work->Vx, 0, &work->lambda, 0);
        double *fun_N = config->cost[N]->memory_get(nlp_mem->cost[N], "fun");

        mem->barrier_lagrangian_curr = 0.0;
        mem->primal_1_curr = 0.0;
        mem->primal_inf = 0.0;
        mem->primal_inf_raw = 0.0;
        mem->cs_inf_mu = 0.0;
        mem->cs_inf_0 = 0.0;
        mem->objective = *fun_N;
        mem->dual_inf = 0.0;
        mem->expected_change_L = 0.0;
        double phi_norm = 0.0;
        double z_norm = 0.0;
        int ni_bounds = 0;

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
            struct blasfeo_dvec *h_scale = mem->h_scale+i;
            struct blasfeo_dvec *g_scale = mem->g_scale+i;
            struct blasfeo_dmat *Hs = ngi > 0 ? &work->Hsolve : &work->H;
            struct blasfeo_dmat *Rs = ngi > 0 ? &work->rhs_u_solve : &work->rhs_u;

            for (int j = 0; j < nui; j++)
                ni_bounds += (VEL(mem->maskul+i, j) != 0.0) + (VEL(mem->maskuu+i, j) != 0.0);
            for (int j = 0; j < ngi; j++)
                ni_bounds += (VEL(mem->maskgl+i, j) != 0.0) + (VEL(mem->maskgu+i, j) != 0.0);

            // linearize stage i with the current dynamics multiplier
            filterddp_set_pi(dims, out, i, unconstrained ? &work->Vx : &work->lambda);
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
                VEL(&work->lx, j) = objective_scale*VEL(cost_grad, nui+j);
            for (int j = 0; j < nui; j++)
                VEL(&work->lu, j) = objective_scale*VEL(cost_grad, j);

            // BAbt = [fu'; fx'; b']
            blasfeo_dgetr(nui, nx[i+1], qp_in->BAbt+i, 0, 0, &work->fu, 0, 0);
            blasfeo_dgetr(nxi, nx[i+1], qp_in->BAbt+i, nui, 0, &work->fx, 0, 0);

            filterddp_gather_constraints(config, dims, in, nlp_opts, nlp_mem, nlp_work, mem, i, &work->h, &work->g);
            filterddp_gather_jacobians(config, dims, nlp_mem, mem, i, work);

            if (nhi > 0)
            {
                for (int j = 0; j < nhi; j++)
                    VEL(&work->h_scaled, j) = VEL(h_scale, j)*VEL(&work->h, j);
                mem->primal_1_curr += filterddp_norm_1(nhi, &work->h_scaled, 0);
                mem->primal_inf = filterddp_max(mem->primal_inf, filterddp_norm_inf(nhi, &work->h_scaled, 0));
                mem->primal_inf_raw = filterddp_max(mem->primal_inf_raw, filterddp_norm_inf(nhi, &work->h, 0));
            }
            if (ngi > 0)
            {
                double q_scaled_1 = 0.0;
                double q_scaled_inf = 0.0;
                for (int j = 0; j < ngi; j++)
                {
                    VEL(&work->q, j) = VEL(&work->g, j) - VEL(s, j);
                    double q_scaled = VEL(g_scale, j)*VEL(&work->q, j);
                    q_scaled_1 += fabs(q_scaled);
                    q_scaled_inf = filterddp_max(q_scaled_inf, fabs(q_scaled));
                }
                mem->primal_1_curr += q_scaled_1;
                mem->primal_inf = filterddp_max(mem->primal_inf, q_scaled_inf);
                mem->primal_inf_raw = filterddp_max(mem->primal_inf_raw, filterddp_norm_inf(ngi, &work->q, 0));
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
                    VEL(&work->sl_dist, j) = VEL(s, j) - VEL(gl, j);
                    VEL(&work->su_dist, j) = VEL(gu_, j) - VEL(s, j);
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

            // Qu = lu + fu' Vx + mu (inv_uu - inv_ul)
            blasfeo_dgemv_t(nxi, nui, 1.0, &work->fu, 0, 0, &work->Vx, 0, 1.0, &work->lu, 0, &work->Qu, 0);
            for (int j = 0; j < nui; j++)
                VEL(&work->Qu, j) += mu*(VEL(&work->inv_uu, j) - VEL(&work->inv_ul, j));
            for (int j = 0; j < ngi; j++)
                VEL(&work->Qs, j) = -VEL(nu_, j) + mu*(VEL(&work->inv_su, j) - VEL(&work->inv_sl, j));

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
            // the cost module scales only its own contribution; apply objective scaling to the cost part by rescaling the gradient above and the Hessian here
            if (objective_scale != 1.0)
            {
                // RSQrq = scale*cost_hess + dyn_hess + constr_hess; the cost contribution is not separable, so rescale the full block
                blasfeo_dgesc(nxi, nxi, objective_scale, &work->C, 0, 0);
                blasfeo_dgesc(nui, nui, objective_scale, &work->H, 0, 0);
                blasfeo_dgesc(nui, nxi, objective_scale, &work->B, 0, 0);
            }

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
                VEL(&work->SigmasL, j) = VEL(&work->inv_sl, j)*VEL(zsl, j);
                VEL(&work->SigmasU, j) = VEL(&work->inv_su, j)*VEL(zsu, j);
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
                for (int j = 0; j < nhi; j++)
                {
                    for (int k = 0; k < nui; k++)
                        EL(&work->hus, j, k) = VEL(h_scale, j)*EL(&work->hu, j, k);
                    for (int k = 0; k < nxi; k++)
                        EL(&work->hxs, j, k) = VEL(h_scale, j)*EL(&work->hx, j, k);
                }
                blasfeo_dgelqf(nhi, nui, &work->hus, 0, 0, &work->lq, 0, 0, work->lq_work);
                blasfeo_dorglq(nui, nui, nhi, &work->lq, 0, 0, &work->Q, 0, 0, work->orglq_work);
                blasfeo_dgetr(nhi, nui, &work->Q, 0, 0, &work->Y, 0, 0);
                if (nz > 0)
                    blasfeo_dgetr(nz, nui, &work->Q, nhi, 0, &work->Z, 0, 0);
                for (int j = 0; j < nhi; j++)
                {
                    EL(&work->aby_tmp, j, 0) = -VEL(&work->h_scaled, j);
                    for (int k = 0; k < nxi; k++)
                        EL(&work->aby_tmp, j, k+1) = -EL(&work->hxs, j, k);
                }
                blasfeo_dgemm_nn(nhi, nhi, nui, 1.0, &work->hus, 0, 0, &work->Y, 0, 0, 0.0, &work->AY, 0, 0, &work->AY, 0, 0);
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
                for (int j = 0; j < nhi; j++)
                    for (int k = 0; k < nxp; k++)
                        EL(mem->psih_omegah+i, j, k) = VEL(h_scale, j)*EL(&work->yr, j, k);
            }
            else
            {
                blasfeo_dtrsm_llnn(nui, nxp, 1.0, &work->Lchol, 0, 0, Rs, 0, 0, &work->sol_tmp, 0, 0);
                blasfeo_dtrsm_lltn(nui, nxp, 1.0, &work->Lchol, 0, 0, &work->sol_tmp, 0, 0, alpha_beta, 0, 0);
            }

            if (ngi > 0)
            {
                blasfeo_dgemm_nn(ngi, nxp, nui, 1.0, &work->gu, 0, 0, alpha_beta, 0, 0, -1.0, &work->rhs_g, 0, 0, mem->alphas_betas+i, 0, 0);
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
                EL(mem->chisl_zetasl+i, j, 0) = VEL(&work->inv_sl, j)*mu - VEL(zsl, j) - VEL(&work->SigmasL, j)*EL(ab_s, j, 0);
                EL(mem->chisu_zetasu+i, j, 0) = VEL(&work->inv_su, j)*mu - VEL(zsu, j) + VEL(&work->SigmasU, j)*EL(ab_s, j, 0);
            }

            // dual infeasibility
            for (int j = 0; j < nui; j++)
                VEL(&work->Lu, j) = VEL(&work->lu, j) - VEL(zl, j) + VEL(zu, j);
            blasfeo_dgemv_t(nxi, nui, 1.0, &work->fu, 0, 0, &work->lambda, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
            if (nhi > 0)
                blasfeo_dgemv_t(nhi, nui, 1.0, &work->hu, 0, 0, phi, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
            if (ngi > 0)
            {
                blasfeo_dgemv_t(ngi, nui, 1.0, &work->gu, 0, 0, nu_, 0, 1.0, &work->Lu, 0, &work->Lu, 0);
                for (int j = 0; j < ngi; j++)
                    VEL(&work->Ls, j) = -VEL(nu_, j) - VEL(zsl, j) + VEL(zsu, j);
                mem->dual_inf = filterddp_max(mem->dual_inf, filterddp_norm_inf(ngi, &work->Ls, 0));
            }
            mem->dual_inf = filterddp_max(mem->dual_inf, filterddp_norm_inf(nui, &work->Lu, 0));
            for (int j = 0; j < nui; j++)
                z_norm += VEL(zl, j) + VEL(zu, j);
            for (int j = 0; j < ngi; j++)
                z_norm += VEL(zsl, j) + VEL(zsu, j);
            for (int j = 0; j < nhi; j++)
                phi_norm += fabs(VEL(phi, j)/VEL(h_scale, j));
            for (int j = 0; j < ngi; j++)
                phi_norm += fabs(VEL(nu_, j)/VEL(g_scale, j));

            // value function recursion
            blasfeo_dgemm_tn(nxi, nxi, nui, 1.0, alpha_beta, 0, 1, &work->B, 0, 0, 1.0, &work->C, 0, 0, &work->Vxx, 0, 0);
            if (nhi > 0)
                blasfeo_dgemm_tn(nxi, nxi, nhi, 1.0, mem->psih_omegah+i, 0, 1, &work->hx, 0, 0, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);
            if (ngi > 0)
                blasfeo_dgemm_tn(nxi, nxi, ngi, 1.0, mem->psig_omegag+i, 0, 1, &work->gx, 0, 0, 1.0, &work->Vxx, 0, 0, &work->Vxx, 0, 0);

            blasfeo_dgemv_t(nui, nxi, 1.0, alpha_beta, 0, 1, &work->Qu, 0, 1.0, &work->lx, 0, &work->Vx_next, 0);
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
            blasfeo_dveccp(nxi, &work->Vx_next, 0, &work->Vx, 0);
            blasfeo_dveccp(nxi, &work->lambda_next, 0, &work->lambda, 0);

            for (int j = 0; j < nui; j++)
                mem->expected_change_L += VEL(&work->Qu, j)*EL(alpha_beta, j, 0);
            for (int j = 0; j < nhi; j++)
                mem->expected_change_L += VEL(&work->h, j)*EL(mem->psih_omegah+i, j, 0);
            for (int j = 0; j < ngi; j++)
                mem->expected_change_L += VEL(&work->Qs, j)*EL(mem->alphas_betas+i, j, 0) + VEL(&work->q, j)*EL(mem->psig_omegag+i, j, 0);
        }
        nlp_timings->time_lin += acados_toc(&timer);

        mem->ni_bounds = ni_bounds;
        double scaling_dual = filterddp_max(opts->s_max, (phi_norm + z_norm)/filterddp_max((double) (ni_bounds + nh_total), 1.0))/opts->s_max;
        double scaling_cs = filterddp_max(opts->s_max, z_norm/filterddp_max((double) ni_bounds, 1.0))/opts->s_max;
        mem->dual_inf /= scaling_dual;
        mem->cs_inf_0 /= scaling_cs;
        mem->cs_inf_mu /= scaling_cs;
        mem->barrier_lagrangian_curr += objective_scale*mem->objective;
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
    const double objective_scale = mem->objective_scale;
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

        for (int j = 0; j < nxi; j++)
            VEL(&work->xi, j) = VEL(trial->ux+i, nui+j) - VEL(out->ux+i, nui+j);

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
        }

        if (!filterddp_all_finite(nui+nxi, trial->ux+i, 0) || !filterddp_all_finite(ngi, mem->s_trial+i, 0))
        {
            mem->status_internal = FILTERDDP_STATUS_FORWARD_FAILED;
            return;
        }

        if (nhi > 0 || ngi > 0)
        {
            filterddp_evaluate_constraints_at(config, dims, in, nlp_opts, nlp_mem, nlp_work, trial, i);
            ocp_nlp_constraints_bgh_memory *constr_mem = nlp_mem->constraints[i];
            ocp_nlp_constraints_bgh_model *model = in->constraints[i];
            int ni = dims->ni[i];
            // recover the raw evaluation from fun = [d_lo - eval; eval - d_up] masked
            for (int j = 0; j < nhi; j++)
            {
                int idx = mem->idxh[i][j];
                double value = VEL(&model->d, idx) - VEL(&constr_mem->fun, idx);
                VEL(&work->h, j) = value;
            }
            for (int j = 0; j < ngi; j++)
            {
                int idx = mem->idxg[i][j];
                double value;
                if (VEL(in->dmask+i, idx) != 0.0)
                    value = VEL(&model->d, idx) - VEL(&constr_mem->fun, idx);
                else
                    value = VEL(&constr_mem->fun, ni+idx) + VEL(&model->d, ni+idx);
                VEL(&work->g, j) = value;
            }
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
                    n1 += fabs(VEL(mem->h_scale+i, j)*VEL(&work->h, j));
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
                    n1 += fabs(VEL(mem->g_scale+i, j)*VEL(&work->q, j));
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
                double sl = VEL(mem->s_trial+i, j) - VEL(mem->gl+i, j);
                double su = VEL(mem->gu+i, j) - VEL(mem->s_trial+i, j);
                double sl_bar = VEL(mem->s+i, j) - VEL(mem->gl+i, j);
                double su_bar = VEL(mem->gu+i, j) - VEL(mem->s+i, j);
                if ((VEL(mem->maskgl+i, j) != 0.0 && sl_bar*one_minus_tau > sl) ||
                    (VEL(mem->maskgu+i, j) != 0.0 && su_bar*one_minus_tau > su))
                {
                    mem->status_internal = FILTERDDP_STATUS_FRACTION_TO_BOUNDARY;
                    return;
                }
            }
            for (int j = 0; j < ngi; j++)
            {
                if (VEL(mem->zsl+i, j)*one_minus_tau > VEL(mem->zsl_trial+i, j) ||
                    VEL(mem->zsu+i, j)*one_minus_tau > VEL(mem->zsu_trial+i, j))
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
                if (VEL(mem->maskgl+i, j) != 0.0)
                    log_l += log(VEL(mem->s_trial+i, j) - VEL(mem->gl+i, j));
                if (VEL(mem->maskgu+i, j) != 0.0)
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
        mem->barrier_lagrangian_next += objective_scale*stage_value;

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
    mem->barrier_lagrangian_next += objective_scale*terminal_value;
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
    }
}



/************************************************
 * output
 ************************************************/

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

    // multipliers of bounds and constraints into the acados layout; bound multipliers are
    // the interior point duals, constraint multipliers phi / nu with sign split over the two sides
    for (int i = 0; i < N; i++)
    {
        int ni = dims->ni[i];
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        filterddp_set_lam(dims, out, mem, i, mem->phi+i, mem->nu+i);
        for (int j = 0; j < cdims->nbu; j++)
        {
            int col = ((ocp_nlp_constraints_bgh_model *) in->constraints[i])->idxb[j];
            VEL(out->lam+i, j) = VEL(mem->zl+i, col);
            VEL(out->lam+i, ni+j) = VEL(mem->zu+i, col);
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            int idx = mem->idxg[i][j];
            VEL(out->lam+i, idx) += VEL(mem->zsl+i, j);
            VEL(out->lam+i, ni+idx) += VEL(mem->zsu+i, j);
        }
        filterddp_set_pi(dims, out, i, i == 0 ? &work->lambda : &work->lambda);
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

    int N = dims->N;

    ocp_nlp_timings_reset(nlp_timings);

#if defined(ACADOS_WITH_OPENMP)
    int num_threads_bkp = omp_get_num_threads();
    omp_set_num_threads(opts->nlp_opts->num_threads);
#endif

    ocp_nlp_initialize_submodules(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);

    if (filterddp_classify_constraints(dims, nlp_in, mem) != ACADOS_SUCCESS)
    {
        nlp_mem->status = ACADOS_QP_FAILURE;
        nlp_timings->time_tot = acados_toc(&timer0);
        return nlp_mem->status;
    }

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_bgh_model *model = nlp_in->constraints[i];
        ocp_nlp_constraints_bgh_dims *cdims = dims->constraints[i];
        int ni = dims->ni[i];
        blasfeo_dvecse(dims->nu[i], -INFINITY, mem->ul+i, 0);
        blasfeo_dvecse(dims->nu[i], INFINITY, mem->uu+i, 0);
        blasfeo_dvecse(dims->nu[i], 0.0, mem->maskul+i, 0);
        blasfeo_dvecse(dims->nu[i], 0.0, mem->maskuu+i, 0);
        for (int j = 0; j < cdims->nbu; j++)
        {
            int col = model->idxb[j];
            if (VEL(nlp_in->dmask+i, j) != 0.0)
            {
                VEL(mem->ul+i, col) = VEL(&model->d, j);
                VEL(mem->maskul+i, col) = 1.0;
            }
            if (VEL(nlp_in->dmask+i, ni+j) != 0.0)
            {
                VEL(mem->uu+i, col) = VEL(&model->d, ni+j);
                VEL(mem->maskuu+i, col) = 1.0;
            }
        }
        for (int j = 0; j < mem->ng[i]; j++)
        {
            int idx = mem->idxg[i][j];
            VEL(mem->maskgl+i, j) = VEL(nlp_in->dmask+i, idx) != 0.0 ? 1.0 : 0.0;
            VEL(mem->maskgu+i, j) = VEL(nlp_in->dmask+i, ni+idx) != 0.0 ? 1.0 : 0.0;
            VEL(mem->gl+i, j) = VEL(mem->maskgl+i, j) != 0.0 ? VEL(&model->d, idx) : -INFINITY;
            VEL(mem->gu+i, j) = VEL(mem->maskgu+i, j) != 0.0 ? VEL(&model->d, ni+idx) : INFINITY;
        }
    }

    filterddp_initialize_trajectory(config, dims, nlp_in, nlp_out, opts, mem, work);

    double previous_objective_scale = mem->objective_scale;
    filterddp_compute_nlp_scaling(config, dims, nlp_in, nlp_out, opts, mem, work);
    double dual_scale = mem->objective_scale/previous_objective_scale;
    if (dual_scale != 1.0)
    {
        for (int i = 0; i < N; i++)
        {
            blasfeo_dvecsc(mem->nh[i], dual_scale, mem->phi+i, 0);
            blasfeo_dvecsc(mem->ng[i], dual_scale, mem->nu+i, 0);
            blasfeo_dvecsc(dims->nu[i], dual_scale, mem->zl+i, 0);
            blasfeo_dvecsc(dims->nu[i], dual_scale, mem->zu+i, 0);
            blasfeo_dvecsc(mem->ng[i], dual_scale, mem->zsl+i, 0);
            blasfeo_dvecsc(mem->ng[i], dual_scale, mem->zsu+i, 0);
        }
    }
    for (int i = 0; i < N; i++)
    {
        blasfeo_dvecsc(dims->nu[i], mem->objective_scale, mem->zl+i, 0);
        blasfeo_dvecsc(dims->nu[i], mem->objective_scale, mem->zu+i, 0);
        blasfeo_dvecsc(mem->ng[i], mem->objective_scale, mem->zsl+i, 0);
        blasfeo_dvecsc(mem->ng[i], mem->objective_scale, mem->zsu+i, 0);
    }

    mem->mu = mem->objective_scale*opts->mu_init;
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

        double opt_err_mu = filterddp_max(filterddp_max(mem->dual_inf, mem->cs_inf_mu), mem->primal_inf);
        double opt_err_0 = filterddp_max(filterddp_max(mem->dual_inf, mem->cs_inf_0), mem->primal_inf);

        if (opt_err_0 < nlp_opts->tol_stat)
        {
            nlp_mem->status = ACADOS_SUCCESS;
            break;
        }
        if (opt_err_mu <= opts->kappa_eps*mem->mu && mem->ni_bounds > 0 && mem->mu > nlp_opts->tol_stat/10.0)
        {
            mem->mu = filterddp_max(nlp_opts->tol_stat/10.0, filterddp_min(opts->kappa_mu*mem->mu, pow(mem->mu, opts->theta_mu)));
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

    if (iter == nlp_opts->max_iter)
        nlp_mem->status = ACADOS_MAXITER;

    filterddp_restore_module_pointers(config, dims, nlp_mem, nlp_out);
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
        else
            printf("\nEXIT: Failed, line-search unable to find acceptable iterate in forward pass.\n\n");
    }

#if defined(ACADOS_WITH_OPENMP)
    omp_set_num_threads(num_threads_bkp);
#endif
    nlp_timings->time_tot = acados_toc(&timer0);

    return nlp_mem->status;
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
        if (dims->ns[i] > 0)
        {
            printf("ocp_nlp_filterddp: soft constraints are not supported, got ns[%d] = %d.\n", i, dims->ns[i]);
            exit(1);
        }
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
    config->opts_get = &ocp_nlp_filterddp_opts_get;
    config->work_get = &ocp_nlp_filterddp_work_get;
    config->terminate = &ocp_nlp_filterddp_terminate;
    config->step_update = &ocp_nlp_filterddp_step_update;
    config->is_real_time_algorithm = &ocp_nlp_filterddp_is_real_time_algorithm;
    config->eval_kkt_residual = &ocp_nlp_filterddp_eval_kkt_residual;

    return;
}
