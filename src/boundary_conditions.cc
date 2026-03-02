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
 * Purpose:    void reapply_temp_gradient:
 *             Re-set temperature Dirichlet BCs at z-faces after every
 *             halo-exchange step.
 *             \f$ T_\mathrm{hot} = N_z + 2 \f$ at \f$ z = 0 \f$ (inlet ghost layer).
 *             \f$ T_\mathrm{cold} = 2 \f$ at \f$ z = L \f$ (outlet ghost layer).
 *             Applied to all voxels (fluid and solid), since heat
 *             conducts through both phases.
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

void reapply_temp_gradient( int* size,
                            int* new_proc_size,
                            double*** temp,
                            int* cart_coords,
                            int* dims)
{
    int j, k;
    int zmax_procs = dims[2] - 1;  // index of last rank in z direction
    int lim = 2;

    if (cart_coords[2] == 0){
        // Set hot-side Dirichlet BC at the inlet ghost layer (i=1).
        // T_hot = size[2] + 2 (mirrors pressure initialisation).
        double T_hot = (double)(size[2] + 2);
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){
                temp[1][j][k] = T_hot;
            }
        }
    }

    if (cart_coords[2] == zmax_procs){
        // Set cold-side Dirichlet BC at the outlet ghost layer.
        // T_cold = 2.0 at outermost ghost (i=new_proc_size[2]-2).
        // Also fix the last interior voxel to 3.0 to maintain gradient.
        for (j = lim; j < new_proc_size[1] - lim; j++){
            for (k = lim; k < new_proc_size[0] - lim; k++){
                temp[new_proc_size[2]-2][j][k] = 2.0;
                temp[new_proc_size[2]-3][j][k] = 3.0;
            }
        }
    }
}
