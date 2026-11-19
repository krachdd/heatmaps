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
 * Purpose:    Boundary conditions of the heat conduction problem.
 *
 *             All BCs are imposed through the two ghost layers of every
 *             sub-domain face that lies on the physical boundary, i.e.
 *             where the Cartesian neighbour is MPI_PROC_NULL. Faces with
 *             a neighbouring rank (or a periodic partner) are filled by
 *             the halo exchange instead.
 *
 *             z faces (inflow/outflow): Dirichlet temperature located on
 *             the outer voxel FACE (half a voxel from the first cell centre).
 *             The ghost cell carries the material of the adjacent cell
 *             (mirror_geom_at_boundaries) and the temperature
 *                 T_ghost = 2 T_bc - T_cell,
 *             so that the face flux is lambda_cell * (T_cell - T_bc) / (dz/2).
 *             With harmonic-mean face conductivities in the interior this
 *             reproduces the series/parallel layer solutions exactly.
 *
 *             x/y faces (only if not periodic): adiabatic (zero flux),
 *             T_ghost = T_cell (mirror).
 *
 *             T_hot  = N_z + 2 at z = 0,  T_cold = 2 at z = L
 *             (Delta T = N_z = L in voxel units).
 *
 * Contents:   mirror_geom_at_boundaries, apply_temp_bcs
 *
 * Changelog:  Version: 1.0.0
 *             krach 2026: Dirichlet values on the boundary face instead of
 *                         ghost-cell centre; ghost material copied from the
 *                         adjacent cell (removes artificial resistive layer);
 *                         no interior layer pinned any more; adiabatic x/y
 *                         walls implemented (previously the uninitialised
 *                         ghost cells acted as T = 0 heat sinks).
 ***********************************************************************/

// HEADER ***************************************************************
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "constants.h"
#include "boundary_conditions.h"
// **********************************************************************


void mirror_geom_at_boundaries( bool*** proc_geom,
                                int* new_proc_size,
                                int* neighbors)
{
    int a, b;
    int nx = new_proc_size[0];
    int ny = new_proc_size[1];
    int nz = new_proc_size[2];

    // z faces first, then y, then x: later directions overwrite the
    // edge/corner ghosts, which are not used by the 7-point stencil anyway.
    for (a = 0; a < ny; a++){
        for (b = 0; b < nx; b++){
            if (neighbors[4] == MPI_PROC_NULL){
                proc_geom[1][a][b] = proc_geom[2][a][b];
                proc_geom[0][a][b] = proc_geom[3][a][b];
            }
            if (neighbors[5] == MPI_PROC_NULL){
                proc_geom[nz-2][a][b] = proc_geom[nz-3][a][b];
                proc_geom[nz-1][a][b] = proc_geom[nz-4][a][b];
            }
        }
    }
    for (a = 0; a < nz; a++){
        for (b = 0; b < nx; b++){
            if (neighbors[2] == MPI_PROC_NULL){
                proc_geom[a][1][b] = proc_geom[a][2][b];
                proc_geom[a][0][b] = proc_geom[a][3][b];
            }
            if (neighbors[3] == MPI_PROC_NULL){
                proc_geom[a][ny-2][b] = proc_geom[a][ny-3][b];
                proc_geom[a][ny-1][b] = proc_geom[a][ny-4][b];
            }
        }
    }
    for (a = 0; a < nz; a++){
        for (b = 0; b < ny; b++){
            if (neighbors[0] == MPI_PROC_NULL){
                proc_geom[a][b][1] = proc_geom[a][b][2];
                proc_geom[a][b][0] = proc_geom[a][b][3];
            }
            if (neighbors[1] == MPI_PROC_NULL){
                proc_geom[a][b][nx-2] = proc_geom[a][b][nx-3];
                proc_geom[a][b][nx-1] = proc_geom[a][b][nx-4];
            }
        }
    }
}


void apply_temp_bcs(int* size,
                    int* new_proc_size,
                    double*** temp,
                    int* neighbors)
{
    int a, b;
    int nx = new_proc_size[0];
    int ny = new_proc_size[1];
    int nz = new_proc_size[2];
    double T_hot  = t_hot(size);
    double T_cold = t_cold(size);

    // x walls: adiabatic (mirror)
    for (a = 0; a < nz; a++){
        for (b = 0; b < ny; b++){
            if (neighbors[0] == MPI_PROC_NULL){
                temp[a][b][1] = temp[a][b][2];
                temp[a][b][0] = temp[a][b][3];
            }
            if (neighbors[1] == MPI_PROC_NULL){
                temp[a][b][nx-2] = temp[a][b][nx-3];
                temp[a][b][nx-1] = temp[a][b][nx-4];
            }
        }
    }
    // y walls: adiabatic (mirror)
    for (a = 0; a < nz; a++){
        for (b = 0; b < nx; b++){
            if (neighbors[2] == MPI_PROC_NULL){
                temp[a][1][b] = temp[a][2][b];
                temp[a][0][b] = temp[a][3][b];
            }
            if (neighbors[3] == MPI_PROC_NULL){
                temp[a][ny-2][b] = temp[a][ny-3][b];
                temp[a][ny-1][b] = temp[a][ny-4][b];
            }
        }
    }
    // z faces: Dirichlet on the boundary face (linear extrapolation into ghost)
    for (a = 0; a < ny; a++){
        for (b = 0; b < nx; b++){
            if (neighbors[4] == MPI_PROC_NULL){
                temp[1][a][b] = 2.0 * T_hot - temp[2][a][b];
                temp[0][a][b] = 2.0 * T_hot - temp[3][a][b];
            }
            if (neighbors[5] == MPI_PROC_NULL){
                temp[nz-2][a][b] = 2.0 * T_cold - temp[nz-3][a][b];
                temp[nz-1][a][b] = 2.0 * T_cold - temp[nz-4][a][b];
            }
        }
    }
}
