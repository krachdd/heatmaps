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
 * Purpose:    double evaluate_heat_flux:
 *             Total z heat flux Q_f through every z face plane
 *             f = 0 (hot boundary face) ... N_z (cold boundary face),
 *             using the same face conductances as the solver
 *             (harmonic mean inside, 2*lambda_cell on the Dirichlet faces).
 *
 *             Effective conductivity (Fourier):
 *             \f[ \lambda_\mathrm{eff} = \frac{\bar{Q}}{N_x N_y}\,\frac{L}{\Delta T},
 *                 \quad \bar{Q} = \frac{1}{N_z+1}\sum_f Q_f \f]
 *
 *             Convergence (returned): in steady state with adiabatic or
 *             periodic side walls every plane carries the same flux, so
 *             \f[ \varepsilon = \max_f \frac{|Q_f - \bar{Q}|}{|\bar{Q}|} \f]
 *             is a residual measure (differences Q_f - Q_{f-1} are the net
 *             heat imbalance of the voxel layers). It does not depend on how
 *             much the field changed since the last check, so a slowly
 *             converging solver cannot stop prematurely. If no heat flows
 *             (Q = 0, e.g. non-percolating conductor) the absolute value
 *             max |Q_f| / (N_x N_y) is returned instead.
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 *             krach 2026: replaced "relative change of the mean flux between
 *                         two checks" by the plane-flux balance above;
 *                         flux evaluated on all faces incl. rank boundaries
 *                         and Dirichlet faces.
 ***********************************************************************/


// HEADER ***************************************************************
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "mpi.h"
#include "constants.h"
#include "evaluation.h"
// **********************************************************************


double evaluate_heat_flux(  bool*** proc_geom,
                            int* size,
                            int* new_proc_size,
                            double*** temp,
                            double cond_solid,
                            double cond_fluid,
                            int* neighbors,
                            int* starts,
                            MPI_Comm comm_cart,
                            double* lambda_eff)
{
    int i, j, k;
    int lim = 2;
    int nx = new_proc_size[0], ny = new_proc_size[1], nz = new_proc_size[2];
    int nfaces = size[2] + 1;
    double T_hot  = t_hot(size);
    double T_cold = t_cold(size);
    bool dir_zm = (neighbors[4] == MPI_PROC_NULL);
    bool dir_zp = (neighbors[5] == MPI_PROC_NULL);

    std::vector<double> Q(nfaces, 0.0);

    // hot boundary face (global face 0), owned by the bottom ranks
    if (dir_zm){
        for (j = lim; j < ny - lim; j++){
            for (k = lim; k < nx - lim; k++){
                double lam_c = (proc_geom[lim][j][k] == 0) ? cond_fluid : cond_solid;
                Q[0] += 2.0 * lam_c * (T_hot - temp[lim][j][k]);
            }
        }
    }

    // face above every local cell layer (global face starts[2] + i - lim + 1);
    // the face to the next rank uses the (current) halo layer
    for (i = lim; i < nz - lim; i++){
        int f = starts[2] + (i - lim) + 1;
        bool top = (i == nz - lim - 1);
        double q = 0.0;
        for (j = lim; j < ny - lim; j++){
            for (k = lim; k < nx - lim; k++){
                double lam_c = (proc_geom[i][j][k] == 0) ? cond_fluid : cond_solid;
                if (top && dir_zp){
                    q += 2.0 * lam_c * (temp[i][j][k] - T_cold);
                }
                else {
                    double lam_n = (proc_geom[i+1][j][k] == 0) ? cond_fluid : cond_solid;
                    q += harm_mean(lam_c, lam_n) * (temp[i][j][k] - temp[i+1][j][k]);
                }
            }
        }
        Q[f] += q;
    }

    MPI_Allreduce(MPI_IN_PLACE, Q.data(), nfaces, MPI_DOUBLE, MPI_SUM, comm_cart);

    double Q_mean = 0.0;
    for (int f = 0; f < nfaces; f++){ Q_mean += Q[f]; }
    Q_mean /= (double)nfaces;

    double area = (double)size[0] * (double)size[1];
    *lambda_eff = Q_mean / area * (double)size[2] / (T_hot - T_cold);

    double dev = 0.0, Qmax = 0.0;
    for (int f = 0; f < nfaces; f++){
        dev  = fmax(dev,  fabs(Q[f] - Q_mean));
        Qmax = fmax(Qmax, fabs(Q[f]));
    }
    // relative imbalance; absolute (per unit area) if (almost) no heat flows
    if (fabs(Q_mean) > 1e-12 * Qmax && fabs(Q_mean) > 0.0){
        return dev / fabs(Q_mean);
    }
    return Qmax / area;
}
