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
 * Purpose:    void heat_update:
 *             One pseudo-time step of the steady-state heat equation
 *             \f$ \nabla\cdot(\lambda\nabla T) = 0 \f$ using a 6-point finite-difference stencil.
 *             Conductivity at each face is computed as the harmonic mean
 *             of the two adjacent voxel conductivities:
 *             \f[ \lambda_\mathrm{face} = \frac{2\,\lambda_A\,\lambda_B}{\lambda_A + \lambda_B} \f]
 *             In-place update with local step 1/sum(lambda_face):
 *             \f[ T \leftarrow T + \omega\,\frac{\sum_f \lambda_f (T_f - T)}{\sum_f \lambda_f} \f]
 *             omega = 1: Gauss-Seidel (solvers 1, 2), omega = omega_sor: SOR (3).
 *
 *             Faces on the physical boundary (neighbour rank MPI_PROC_NULL)
 *             are treated implicitly inside the stencil, not via ghost cells:
 *               x/y walls : lambda_face = 0 (adiabatic)
 *               z faces   : lambda_face = 2 lambda_cell, T_f = T_hot / T_cold
 *                           (Dirichlet value on the boundary face, half a
 *                           voxel from the cell centre)
 *             This keeps the discrete operator symmetric positive definite,
 *             so SOR converges for 0 < omega < 2. (Lagged ghost-cell BCs made
 *             SOR diverge for omega >= 1.8.)
 *
 *             Red-black ordering (argument colour = 0 or 1, by global index
 *             parity); calc_heat exchanges halos after each colour. Without it
 *             the lagged inter-rank halos turned SOR into block-SOR, which
 *             diverged on more than one rank.
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


void heat_update(   bool*** proc_geom,
                    int* size,
                    int* new_proc_size,
                    int* neighbors,
                    int* starts,
                    int colour,
                    double*** temp,
                    double cond_solid,
                    double cond_fluid,
                    unsigned int solver)
{
    int i, j, k;
    int lim = 2;
    int nx = new_proc_size[0], ny = new_proc_size[1], nz = new_proc_size[2];
    double omega  = (solver == 3) ? omega_sor : 1.0;
    double T_hot  = t_hot(size);
    double T_cold = t_cold(size);

    // physical boundary faces of this sub-domain
    bool wall_xm = (neighbors[0] == MPI_PROC_NULL), wall_xp = (neighbors[1] == MPI_PROC_NULL);
    bool wall_ym = (neighbors[2] == MPI_PROC_NULL), wall_yp = (neighbors[3] == MPI_PROC_NULL);
    bool dir_zm  = (neighbors[4] == MPI_PROC_NULL), dir_zp  = (neighbors[5] == MPI_PROC_NULL);

    // Red-black ordering: only voxels with (global i+j+k) % 2 == colour are
    // updated. Same-colour voxels are not coupled by the 7-point stencil, so a
    // colour sweep is order-independent and, with a halo exchange between the
    // two colours, the parallel iteration equals the serial one exactly.
    int parity0 = (starts[0] + starts[1] + starts[2]) % 2;

    for (i = lim; i < nz - lim; i++){
        for (j = lim; j < ny - lim; j++){
            int k0 = lim + ((parity0 + (i - lim) + (j - lim) + colour) % 2);
            for (k = k0; k < nx - lim; k += 2){

                // Conductivity at current voxel
                double lam_c = (proc_geom[i][j][k] == 0) ? cond_fluid : cond_solid;

                // Conductivities at the 6 neighbours (z-, z+, y-, y+, x-, x+)
                double lam_zm_n = (proc_geom[i-1][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_zp_n = (proc_geom[i+1][j][k] == 0) ? cond_fluid : cond_solid;
                double lam_ym_n = (proc_geom[i][j-1][k] == 0) ? cond_fluid : cond_solid;
                double lam_yp_n = (proc_geom[i][j+1][k] == 0) ? cond_fluid : cond_solid;
                double lam_xm_n = (proc_geom[i][j][k-1] == 0) ? cond_fluid : cond_solid;
                double lam_xp_n = (proc_geom[i][j][k+1] == 0) ? cond_fluid : cond_solid;

                // Harmonic-mean face conductivities
                double lam_zm = harm_mean(lam_c, lam_zm_n);
                double lam_zp = harm_mean(lam_c, lam_zp_n);
                double lam_ym = harm_mean(lam_c, lam_ym_n);
                double lam_yp = harm_mean(lam_c, lam_yp_n);
                double lam_xm = harm_mean(lam_c, lam_xm_n);
                double lam_xp = harm_mean(lam_c, lam_xp_n);

                // Neighbour temperatures
                double T_zm = temp[i-1][j][k], T_zp = temp[i+1][j][k];
                double T_ym = temp[i][j-1][k], T_yp = temp[i][j+1][k];
                double T_xm = temp[i][j][k-1], T_xp = temp[i][j][k+1];

                // Physical boundary faces
                if (wall_xm && k == lim)      { lam_xm = 0.0; }
                if (wall_xp && k == nx-lim-1) { lam_xp = 0.0; }
                if (wall_ym && j == lim)      { lam_ym = 0.0; }
                if (wall_yp && j == ny-lim-1) { lam_yp = 0.0; }
                if (dir_zm  && i == lim)      { lam_zm = 2.0 * lam_c; T_zm = T_hot;  }
                if (dir_zp  && i == nz-lim-1) { lam_zp = 2.0 * lam_c; T_zp = T_cold; }

                // Weighted Laplacian residual
                double delta =
                    lam_zm * (T_zm - temp[i][j][k]) +
                    lam_zp * (T_zp - temp[i][j][k]) +
                    lam_ym * (T_ym - temp[i][j][k]) +
                    lam_yp * (T_yp - temp[i][j][k]) +
                    lam_xm * (T_xm - temp[i][j][k]) +
                    lam_xp * (T_xp - temp[i][j][k]);

                // Local pseudo-time step 1/sum(lambda_face)
                double lam_sum = lam_zm + lam_zp + lam_ym + lam_yp + lam_xm + lam_xp;
                if (lam_sum > 0.0){
                    temp[i][j][k] += omega * delta / lam_sum;
                }
            }
        }
    }
}
