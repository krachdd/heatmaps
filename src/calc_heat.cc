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
 * Purpose:    void calc_heat:
 *             Main loop: update T → communicate halos →
 *             apply boundary ghost layers → check convergence.
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
#include "parallelization.h"
#include "solver.h"
#include "boundary_conditions.h"
#include "evaluation.h"
#include "output.h"
// **********************************************************************

void calc_heat( bool*** proc_geom,
                int* size,
                int* new_proc_size,
                unsigned int* bc,
                double* dx,
                int max_iterations,
                double*** temp,
                int it_write,
                int it_eval,
                unsigned int solver,
                double eps,
                double cond_solid,
                double cond_fluid,
                int n_proc,
                int rank,
                MPI_Comm comm_cart,
                int* dims,
                int* cart_coords,
                int* neighbors,
                int* starts,
                int* ends,
                char* log_file)
{

    int iteration = 0;                           // current iteration counter
    double flux_conv = 1.0 + eps;                // convergence criterion (initialised above threshold)
    double start_it0, end_it0, elapsed_seconds;  // per-iteration timing (rank 0 only)
    double lambda_eff = 0.0;                      // effective thermal conductivity
    bool   firstline  = true;                     // controls whether to write the header line

    if (rank == 0){printf("START MAIN COMPUTATION\n");}

    while (iteration < max_iterations && flux_conv >= eps){

        if (rank == 0){start_it0 = MPI_Wtime();}

        if (iteration == 0){
            // Set initial Dirichlet BCs before first solver step
            apply_temp_bcs(size, new_proc_size, temp, neighbors);
        }

        // One iteration = red sweep + black sweep, halo exchange after each
        for (int colour = 0; colour < 2; colour++){
            heat_update(proc_geom, size, new_proc_size, neighbors, starts, colour,
                        temp, cond_solid, cond_fluid, solver);

            // Communicate temperature halos between ranks
            communicate_halos(temp, new_proc_size, neighbors, MPI_DOUBLE, comm_cart, bc, 0);

            // Boundary ghost layers (output only; the solver treats boundary
            // faces inside the stencil): Dirichlet z faces, adiabatic x/y walls
            apply_temp_bcs(size, new_proc_size, temp, neighbors);
        }

        // Evaluate convergence (plane-flux balance) and write logfile
        if (iteration % it_eval == 0){
            flux_conv = evaluate_heat_flux( proc_geom, size, new_proc_size, temp,
                                            cond_solid, cond_fluid, neighbors, starts,
                                            comm_cart, &lambda_eff);

            bool last = (flux_conv < eps) || (iteration + it_eval >= max_iterations);
            if (rank == 0 && (iteration % it_write == 0 || last)){
                end_it0 = MPI_Wtime(); elapsed_seconds = end_it0 - start_it0;
                write_logfile(  iteration,
                                lambda_eff,
                                flux_conv,
                                log_file,
                                firstline,
                                (1.0/elapsed_seconds));
                firstline = false;
            }
        }

        if (rank == 0 && iteration % 100 == 0){
            end_it0 = MPI_Wtime(); elapsed_seconds = end_it0 - start_it0;
            printf("Iteration: %i\t Convergence: %e\t", iteration, flux_conv);
            printf("Time: %f\t TPS: %f\n", elapsed_seconds, (1.0/elapsed_seconds));
        }

        iteration++;
    }

    if (rank == 0){
        if (flux_conv < eps){
            printf("Converged after %i iterations: plane-flux imbalance %e < eps = %e\n",
                   iteration, flux_conv, eps);
        }
        else {
            printf("NOT converged: max_iter = %i reached, plane-flux imbalance %e >= eps = %e\n",
                   max_iterations, flux_conv, eps);
        }
        printf("lambda_eff = %.10e\n", lambda_eff);
    }
}
