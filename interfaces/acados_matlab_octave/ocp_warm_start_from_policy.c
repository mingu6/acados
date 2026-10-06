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

// system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// acados
#include "acados_c/ocp_nlp_interface.h"
// mex
#include "mex.h"


// status = ocp_warm_start_from_policy(C_ocp, x0): FILTERDDP warm start of the next solve from the policy of the last
// solve, rolled out in closed loop from x0, see ocp_nlp_warm_start_from_policy
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    long long *ptr;

    /* RHS */

    // C_ocp
    ptr = (long long *) mxGetData( mxGetField( prhs[0], 0, "config" ) );
    ocp_nlp_config *config = (ocp_nlp_config *) ptr[0];
    ptr = (long long *) mxGetData( mxGetField( prhs[0], 0, "dims" ) );
    ocp_nlp_dims *dims = (ocp_nlp_dims *) ptr[0];
    ptr = (long long *) mxGetData( mxGetField( prhs[0], 0, "in" ) );
    ocp_nlp_in *in = (ocp_nlp_in *) ptr[0];
    ptr = (long long *) mxGetData( mxGetField( prhs[0], 0, "out" ) );
    ocp_nlp_out *out = (ocp_nlp_out *) ptr[0];
    ptr = (long long *) mxGetData( mxGetField( prhs[0], 0, "solver" ) );
    ocp_nlp_solver *solver = (ocp_nlp_solver *) ptr[0];

    // x0
    int nx0 = ocp_nlp_dims_get_from_attr(config, dims, out, 0, "x");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || (int) mxGetNumberOfElements(prhs[1]) != nx0)
    {
        char buffer[200];
        snprintf(buffer, sizeof(buffer),
            "ocp_warm_start_from_policy: x0 must be a real double vector with %d entries, got %d.",
            nx0, (int) mxGetNumberOfElements(prhs[1]));
        mexErrMsgTxt(buffer);
    }
    double *x0 = mxGetPr(prhs[1]);

    /* solver */
    int status = ocp_nlp_warm_start_from_policy(solver, in, out, x0);

    /* LHS */
    plhs[0] = mxCreateDoubleScalar((double) status);

    return;
}
