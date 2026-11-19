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
 * Purpose:    constants for heat conduction solver
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 *
 * file_name_char  : max bytes for a filename string
 * pass            : no-op placeholder macro
 * omega_sor       : over-relaxation factor for solver 3 (SOR), 1 < omega < 2;
 *                   solvers 1/2 use omega = 1 (Gauss-Seidel)
 * ndims           : number of spatial dimensions of the Cartesian grid
 * t_hot / t_cold  : Dirichlet temperatures at z = 0 / z = L
 *                   (T_hot - T_cold = N_z, i.e. unit gradient in voxel units)
 *
 *             krach 2026: removed dt_heat. The update now uses the local
 *                         step 1/sum(lambda_face) (exact Gauss-Seidel), which
 *                         is stable for any conductivities. The old fixed
 *                         dt_heat = 1e-4 was ~1000x below the stability limit.
 ***********************************************************************/

#ifndef file_name_char
    #define file_name_char 400
#endif
#ifndef pass
    #define pass (void)0
#endif
#ifndef omega_sor
    #define omega_sor 1.9
#endif
#ifndef ndims
    #define ndims 3
#endif

// Harmonic mean of two face conductivities; 0 if either side is 0.
static inline double harm_mean(double a, double b){ return (a + b > 0.0) ? 2.0 * a * b / (a + b) : 0.0; }
static inline double t_hot(const int* size){ return (double)size[2] + 2.0; }
static inline double t_cold(const int* size){ (void)size; return 2.0; }
