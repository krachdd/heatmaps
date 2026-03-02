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
 * Purpose:    Header for solver.cc
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 *
 * Solver:    1: Jacobi (same as 2 for single-field heat)
 *            2: Gauss-Seidel (uses updated T values in same sweep)
 *            3: SOR (successive over-relaxation)
 ***********************************************************************/


void heat_update(   bool*** proc_geom,
                    int* new_proc_size,
                    double*** temp,
                    double cond_solid,
                    double cond_fluid,
                    unsigned int solver);
