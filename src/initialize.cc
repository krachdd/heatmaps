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
 * Purpose:    void allocate_initial_field:
 *             Allocate and initialise temperature field with a linear
 *             z-gradient \f$ T_{i,j,k}^0 = (N_z - i_\mathrm{global}) + 2 \f$,
 *             then communicate halos.
 *
 *             void determine_comm_pattern:
 *             Set per-rank bc[] array that controls which directions
 *             communicate (MPI) and which apply a wall condition.
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 *
 * Each rank stores bc[3], one entry per spatial direction [x, y, z].
 * Values encode which wall (if any) the rank borders and the wall type:
 *
 *   0:  communicate in both directions (interior rank)
 *   1:  communicate in +1, slip condition in -1
 *   2:  communicate in +1, no-slip condition in -1
 *   3:  communicate in -1, slip condition in +1
 *   4:  communicate in -1, no-slip condition in +1
 *   5:  slip condition in both -1 and +1
 *   6:  no-slip condition in both -1 and +1
 *
 ************************************************************************/


// HEADER ***************************************************************
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <math.h>
#include "mpi.h"
#include "constants.h"
#include "parallelization.h"
// **********************************************************************


void allocate_initial_field(int* size,
                            int* new_proc_size,
                            double*** temp,
                            MPI_Comm comm_cart,
                            int* neighbors,
                            int* starts,
                            unsigned int* bc)
{
    int i, j, k;

    // Allocate temperature array
    for (i = 0; i < new_proc_size[2]; i++){
        temp[i] = (double**)malloc(new_proc_size[1]*sizeof(double*));
        for (j = 0; j < new_proc_size[1]; j++){
            temp[i][j] = (double*)malloc(new_proc_size[0]*sizeof(double));
            for (k = 0; k < new_proc_size[0]; k++){
                temp[i][j][k] = 0.0;
            }
        }
    }

    // Initialise temperature with a linear gradient along z:
    // T[i][j][k] = (N_z - i_global) + 2, matching the Dirichlet BCs
    // T_hot = N_z + 2 at z=0 and T_cold = 2 at z=L.
    // Applied to all voxels (fluid and solid).
    int lim = 2;
    for (i = lim; i < new_proc_size[2] - lim; i++){
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){
                temp[i][j][k] = (size[2] - starts[2] - i) + lim + 2;
            }
        }
    }

    // Communicate temperature values into halo layers.
    communicate_halos(temp, new_proc_size, neighbors, MPI_DOUBLE, comm_cart, bc, 0);
}


void determine_comm_pattern(int* dims,
                            int rank,
                            int* cart_coords,
                            unsigned int* bc,
                            int bc_method,
                            int* neighbors)
{
/*
 * bc_method selects the physical setup:
 *
 *   0:  fully periodic in all directions
 *
 *   1:  periodic in z; slip walls in x and y
 *
 *   2:  periodic in z; no-slip walls in x and y
 *
 *   3:  non-periodic; slip walls in x and y; temperature-driven in z
 *
 *   4:  non-periodic; no-slip walls in x and y; temperature-driven in z
 *       (default for heat conduction benchmark)
 */

    int i;

    // Default: all ranks communicate in both directions.
    for (i = 0; i < ndims; i++){ bc[i] = 0; }

    if (bc_method == 0){
        // Fully periodic — bc stays 0 for all ranks and all directions.
    }

    else if (bc_method == 1){
        bc[2] = 0;

        if (dims[0] == 1){ bc[0] = 5; }
        else if (dims[0] > 1){
            if (cart_coords[0] == 0)           { bc[0] = 1; }
            if (cart_coords[0] + 1 == dims[0]) { bc[0] = 3; }
        }

        if (dims[1] == 1){ bc[1] = 5; }
        else if (dims[1] > 1){
            if (cart_coords[1] == 0)           { bc[1] = 1; }
            if (cart_coords[1] + 1 == dims[1]) { bc[1] = 3; }
        }
    }

    else if (bc_method == 2){
        bc[2] = 0;

        if (dims[0] == 1){ bc[0] = 6; }
        else if (dims[0] > 1){
            if (cart_coords[0] == 0)           { bc[0] = 2; }
            if (cart_coords[0] + 1 == dims[0]) { bc[0] = 4; }
        }

        if (dims[1] == 1){ bc[1] = 6; }
        else if (dims[1] > 1){
            if (cart_coords[1] == 0)           { bc[1] = 2; }
            if (cart_coords[1] + 1 == dims[1]) { bc[1] = 4; }
        }
    }

    else if (bc_method == 3){
        if (dims[0] == 1){ bc[0] = 5; }
        else if (dims[0] > 1){
            if (cart_coords[0] == 0)           { bc[0] = 1; }
            if (cart_coords[0] + 1 == dims[0]) { bc[0] = 3; }
        }

        if (dims[1] == 1){ bc[1] = 5; }
        else if (dims[1] > 1){
            if (cart_coords[1] == 0)           { bc[1] = 1; }
            if (cart_coords[1] + 1 == dims[1]) { bc[1] = 3; }
        }

        if (dims[2] == 1){ bc[2] = 5; }
        else if (dims[2] > 1){
            if (cart_coords[2] == 0)           { bc[2] = 1; }
            if (cart_coords[2] + 1 == dims[2]) { bc[2] = 3; }
        }
    }

    else if (bc_method == 4){
        // Non-periodic; no-slip x and y walls; Dirichlet T in z.
        if (dims[0] == 1){ bc[0] = 6; }
        else if (dims[0] > 1){
            if (cart_coords[0] == 0)           { bc[0] = 2; }
            if (cart_coords[0] + 1 == dims[0]) { bc[0] = 4; }
        }

        if (dims[1] == 1){ bc[1] = 6; }
        else if (dims[1] > 1){
            if (cart_coords[1] == 0)           { bc[1] = 2; }
            if (cart_coords[1] + 1 == dims[1]) { bc[1] = 4; }
        }

        // z direction: Dirichlet BCs are applied by reapply_temp_gradient,
        // not by halo communication. Use slip-like values here so that
        // communicate_halos does not overwrite the Dirichlet ghost layers.
        if (dims[2] == 1){ bc[2] = 5; }
        else if (dims[2] > 1){
            if (cart_coords[2] == 0)           { bc[2] = 2; }
            if (cart_coords[2] + 1 == dims[2]) { bc[2] = 4; }
        }
    }
}
