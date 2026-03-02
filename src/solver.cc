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
 *             Update (Gauss-Seidel / Jacobi):
 *             \f[ T_{i,j,k}^{n+1} = T_{i,j,k}^{n} + \Delta t_\mathrm{heat}\,\mathcal{R}_{i,j,k} \f]
 *             where \f$ \mathcal{R}_{i,j,k} \f$ is the weighted Laplacian residual.
 *             SOR applies an additional over-relaxation factor \f$ \omega \f$:
 *             \f[ T_{i,j,k}^{n+1} = T_{i,j,k}^{n} + \omega\,\Delta t_\mathrm{heat}\,\mathcal{R}_{i,j,k} \f]
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
                    int* new_proc_size,
                    double*** temp,
                    double cond_solid,
                    double cond_fluid,
                    unsigned int solver)
{
    int i, j, k;
    int lim = 2;

    for (i = lim; i < new_proc_size[2] - lim; i++){
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){

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
                double lam_zm = 2.0 * lam_c * lam_zm_n / (lam_c + lam_zm_n);
                double lam_zp = 2.0 * lam_c * lam_zp_n / (lam_c + lam_zp_n);
                double lam_ym = 2.0 * lam_c * lam_ym_n / (lam_c + lam_ym_n);
                double lam_yp = 2.0 * lam_c * lam_yp_n / (lam_c + lam_yp_n);
                double lam_xm = 2.0 * lam_c * lam_xm_n / (lam_c + lam_xm_n);
                double lam_xp = 2.0 * lam_c * lam_xp_n / (lam_c + lam_xp_n);

                // Weighted Laplacian residual
                double delta =
                    lam_zm * (temp[i-1][j][k] - temp[i][j][k]) +
                    lam_zp * (temp[i+1][j][k] - temp[i][j][k]) +
                    lam_ym * (temp[i][j-1][k] - temp[i][j][k]) +
                    lam_yp * (temp[i][j+1][k] - temp[i][j][k]) +
                    lam_xm * (temp[i][j][k-1] - temp[i][j][k]) +
                    lam_xp * (temp[i][j][k+1] - temp[i][j][k]);

                if (solver == 1 || solver == 2){
                    temp[i][j][k] += dt_heat * delta;
                }
                else if (solver == 3){
                    temp[i][j][k] += omega_sor * dt_heat * delta;
                }
            }
        }
    }
}
