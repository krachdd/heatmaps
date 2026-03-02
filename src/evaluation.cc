/************************************************************************

Parallel Finite Difference Solver for Heat Conduction in Porous Media
Copyright 2026 David Krach

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

 * Authors:    David Krach
 * Date:       2024
 * Contact:    david.krach@mib.uni-stuttgart.de
 *
 * Purpose:    void compute_eff_conductivity:
 *             Compute effective thermal conductivity from mean z-flux.
 *             Analogy to Darcy permeability (Fourier's law):
 *             \f[ \lambda_\mathrm{eff} = \bar{q}_z \cdot \frac{L}{\Delta T} \f]
 *             where \f$ \bar{q}_z = \frac{1}{N}\sum_{i,j,k}
 *             \lambda_\mathrm{face}\,(T_{i,j,k} - T_{i+1,j,k}) \f$
 *             over all interior voxel pairs, \f$ L = N_z \f$,
 *             \f$ \Delta T = T_\mathrm{hot} - T_\mathrm{cold} = N_z \f$.
 *             Since \f$ L/\Delta T = 1 \f$, \f$ \lambda_\mathrm{eff} = \bar{q}_z \f$
 *             in simulation units (same as input conductivities).
 *
 *             double compute_convergence:
 *             Relative change in mean z-flux between evaluation steps:
 *             \f[ \varepsilon = \frac{|\bar{q}_z^{(n)} - \bar{q}_z^{(n-1)}|}{|\bar{q}_z^{(n)}|} \f]
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 ***********************************************************************/


// HEADER ***************************************************************
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "constants.h"
// **********************************************************************


void compute_eff_conductivity(  bool*** proc_geom,
                                int* size,
                                int* new_proc_size,
                                double*** temp,
                                double* lambda_eff,
                                double cond_solid,
                                double cond_fluid,
                                MPI_Comm comm_cart,
                                double voxelsize)
{
    int i, j, k;
    int lim = 2;
    double flux_sum = 0.0;
    double vox_count = 0.0;

    // Sum the z-direction heat flux over all interior voxels.
    // flux at face between (i,j,k) and (i+1,j,k) uses harmonic-mean conductivity.
    // We skip the last interior layer in z (i=new_proc_size[2]-lim-1) to avoid
    // reading the ghost layer at i+1, but include it in vox_count.
    for (i = lim; i < new_proc_size[2] - lim - 1; i++){
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){
                double lam_c = (proc_geom[i][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_n = (proc_geom[i+1][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_face = 2.0 * lam_c * lam_n / (lam_c + lam_n);
                // Heat flux: q_z = -λ dT/dz ≈ λ_face*(T[i] - T[i+1]) > 0
                // (T decreases with i, so T[i] - T[i+1] > 0 gives positive flux)
                flux_sum += lam_face * (temp[i][j][k] - temp[i+1][j][k]);
                vox_count += 1.0;
            }
        }
    }

    // Reduce across all ranks
    double gbuf[2] = {flux_sum, vox_count};
    MPI_Allreduce(MPI_IN_PLACE, gbuf, 2, MPI_DOUBLE, MPI_SUM, comm_cart);
    flux_sum  = gbuf[0];
    vox_count = gbuf[1];

    // mean_flux_z = sum / N_voxels
    double mean_flux_z = flux_sum / vox_count;

    // λ_eff = mean_flux_z * (L / ΔT); with L = ΔT = N_z the factor L/ΔT = 1.
    // Result is in the same units as cond_solid/cond_fluid (W/(m·K) if physical).
    *lambda_eff = mean_flux_z;
}


double compute_convergence( bool*** proc_geom,
                            int* size,
                            int* new_proc_size,
                            double*** temp,
                            double* flux_prev,
                            double cond_solid,
                            double cond_fluid,
                            MPI_Comm comm_cart)
{
    int i, j, k;
    int lim = 2;
    double flux_sum = 0.0;
    double vox_count = 0.0;
    double conv = 0.0;

    for (i = lim; i < new_proc_size[2] - lim - 1; i++){
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){
                double lam_c = (proc_geom[i][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_n = (proc_geom[i+1][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_face = 2.0 * lam_c * lam_n / (lam_c + lam_n);
                flux_sum += lam_face * (temp[i][j][k] - temp[i+1][j][k]);
                vox_count += 1.0;
            }
        }
    }

    // Batch both reductions into one call
    double cbuf[2] = {flux_sum, vox_count};
    MPI_Allreduce(MPI_IN_PLACE, cbuf, 2, MPI_DOUBLE, MPI_SUM, comm_cart);
    flux_sum  = cbuf[0];
    vox_count = cbuf[1];

    double current_flux = flux_sum / vox_count;

    // Convergence: relative change in mean flux
    if (fabs(current_flux) > 0.0){
        conv = fabs(current_flux - *flux_prev) / fabs(current_flux);
    }
    else {
        conv = 0.0;
    }
    *flux_prev = current_flux;

    return conv;
}
