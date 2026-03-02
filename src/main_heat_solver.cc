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
 * Purpose:    Main Heat-Conduction Solver File.
 *             Solves the steady-state heat equation ∇·(λ∇T)=0 in a
 *             3D porous domain using pseudo-time relaxation.
 *             The scalar temperature field T replaces the four Stokes
 *             fields (p, vx, vy, vz). Effective thermal conductivity
 *             λ_eff is the scalar output analogous to permeability.
 *
 * Contents:
 *
 * Changelog:  Version: 1.0.0
 ***********************************************************************/


// HEADER ***************************************************************
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include "mpi.h"
#include "read_problem_params.h"
#include "constants.h"
#include "geometry.h"
#include "output.h"
#include "calc_heat.h"
#include "parallelization.h"
#include "initialize.h"
//**********************************************************************

int main (int argc, char** argv){
    /***********************************************************************
     * Declare all variables
    ************************************************************************/
    // Parallelization
    int           rank, n_proc;
    int           rank_glob;
    MPI_Comm      comm_cart;

    int           dims[ndims];
    int           periods[ndims];
    const int     reorder = true;
    int           neighbors[2*ndims];
    int           cart_coords[ndims];

    // Parametrization
    int           i;
    double        eps;
    int           max_iterations;
    int           it_eval, it_write;
    unsigned int  solver;

    // File names
    std::string   input_fn;
    char          input_filename[file_name_char];
    char          GEOM_FILE[file_name_char];
    char          temp_filename[file_name_char] = "temp_";
    char          domd_filename[file_name_char] = "domain_decomp_";
    char          log_file[file_name_char];
    int           write_output[2];   // [0]=temp, [1]=dom_decomp

    // Problem variables
    double        dx;
    double        cond_solid;        // thermal conductivity of solid phase
    double        cond_fluid;        // thermal conductivity of fluid phase
    int           size[3];
    bool          ***proc_geometry;
    int           ***domain_decomposition = nullptr;
    unsigned int  bc[3];
    int           bc_method;
    int           proc_size[3];
    int           new_proc_size[3];
    int           starts[3], ends[3];
    int           dom_interest[2*ndims];

    // Field variable
    double        ***temp;


    /***********************************************************************
     * End of Declare all variables
    ************************************************************************/

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_glob);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    if (rank_glob == 0){
        std::time_t starttime = std::time(nullptr);
        std::cout << "\nStarting Heat Conduction Solver at " << std::asctime(std::localtime(&starttime));

        std::cout << "\nFinite Difference Heat Conduction Solver  Copyright (C) 2026  David Krach" << std::endl;
        std::cout << "This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.md." << std::endl;
        std::cout << "This is free software, and you are welcome to redistribute it" << std::endl;
        std::cout << "under certain conditions; see LICENSE.md for details.\n" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Define input file
    if (argc == 1){input_fn = "input_param.inp";}
    else if (argc == 2){input_fn = (argv[1]);}

    if (rank_glob == 0){
        if (argc == 1){printf("No input file specified. Using input_param.inp\n");}
        else if (argc == 2){printf("Input file: %s\n", input_fn.c_str());}

        if (FILE *file = fopen(input_fn.c_str(), "r")) {fclose(file);}
        else {fprintf(stderr, "Input file doesnt exist.\n");}
    }

    memcpy(input_filename, input_fn.c_str(), input_fn.size()+1);

    // Read parallel params first (needed to set up Cartesian topology)
    read_parallel_params(input_filename, &bc_method, dims, rank_glob);
    if (bc_method == 0){                  periods[0] = 1; periods[1] = 1; periods[2] = 1;}
    if (bc_method == 1 || bc_method == 2){periods[0] = 0; periods[1] = 0; periods[2] = 1;}
    if (bc_method == 3 || bc_method == 4){periods[0] = 0; periods[1] = 0; periods[2] = 0;}

    if (dims[0] == 0 && dims[1] == 0 && dims[2] == 0){
        for (i = 0; i < ndims; i++){dims[i] = 0;}
        MPI_Dims_create(n_proc, ndims, dims);
        if (rank_glob == 0){
            printf("MPI domain decomposition nx = %i, ny = %i, nz = %i\n", dims[0], dims[1], dims[2]);
        }
    }
    else {
        if (rank_glob == 0){
            printf("Given domain decomposition nx = %i, ny = %i, nz = %i\n", dims[0], dims[1], dims[2]);
        }
    }

    // Create Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, ndims, cart_coords);

    printf("Rank %i reports --> Coordinates: cx = %i, cy = %i, cz = %i\n",
           rank, cart_coords[0], cart_coords[1], cart_coords[2]);

    for (i = 0; i < ndims; i++){
        MPI_Cart_shift(comm_cart, i, 1, &neighbors[i*2], &neighbors[i*2+1]);
    }

    // Read all problem parameters
    read_problem_params(    input_filename,
                            GEOM_FILE,
                            size,
                            &dx,
                            &max_iterations,
                            &it_eval,
                            &it_write,
                            log_file,
                            &solver,
                            &eps,
                            &cond_solid,
                            &cond_fluid,
                            dom_interest,
                            write_output,
                            rank);

    // Compute domain limits
    get_dom_limits(size, proc_size, new_proc_size, starts, ends, dims, rank, cart_coords);
    // Determine communication pattern based on BC method
    determine_comm_pattern(dims, rank, cart_coords, bc, bc_method, neighbors);

    printf("Rank %i: Coordinates: cx = %i, cy = %i, cz = %i\n",
                rank, cart_coords[0], cart_coords[1], cart_coords[2]);
    printf("Rank %i: Domain size: domx = %i, domy = %i, domz = %i\n",
                rank, proc_size[0], proc_size[1], proc_size[2]);
    printf("Rank %i: Full domain size incl. halos: domx = %i, domy = %i, domz = %i\n",
                rank, new_proc_size[0], new_proc_size[1], new_proc_size[2]);
    printf("Rank %i: Domain limits: x_lims (%i, %i), y_lims (%i, %i), z_lims (%i, %i)\n",
                rank, starts[0], ends[0], starts[1], ends[1], starts[2], ends[2]);
    printf("Rank %i: Neighbors: left = %i, right = %i, top = %i, bottom = %i, front = %i, back = %i\n",
                rank, neighbors[0], neighbors[1], neighbors[2], neighbors[3], neighbors[4], neighbors[5]);
    printf("Rank %i: Boundary Conditions: x : %i, y : %i, z : %i\n",
                rank, bc[0], bc[1], bc[2]);
    MPI_Barrier(comm_cart);

    // Allocate arrays
    proc_geometry = (bool***)malloc((new_proc_size[2])*sizeof(bool**));
    temp          = (double***)malloc((new_proc_size[2])*sizeof(double**));
    if (write_output[1] == 1){
        domain_decomposition = (int***)malloc((new_proc_size[2])*sizeof(int**));
    }

    // Read input geometry
    read_geometry(  GEOM_FILE,
                    size,
                    proc_size,
                    new_proc_size,
                    starts,
                    proc_geometry,
                    comm_cart,
                    dims,
                    rank,
                    n_proc,
                    cart_coords,
                    neighbors);

    MPI_Barrier(comm_cart);

    // Allocate temperature field and set initial conditions
    allocate_initial_field( size,
                            new_proc_size,
                            temp,
                            comm_cart,
                            neighbors,
                            starts,
                            bc);

    // Main heat computation loop
    calc_heat(  proc_geometry,
                size,
                new_proc_size,
                bc,
                &dx,
                max_iterations,
                temp,
                it_write,
                it_eval,
                solver,
                eps,
                cond_solid,
                cond_fluid,
                n_proc,
                rank,
                comm_cart,
                dims,
                cart_coords,
                neighbors,
                starts,
                ends,
                log_file);

    // Build output file names by prepending prefix to geometry file name
    strcat(temp_filename, GEOM_FILE);
    strcat(domd_filename, GEOM_FILE);

    if (rank == 0){printf("Writing output to file\n");}

    if (write_output[0] == 1){
        if (rank == 0){printf("Writing temperature field to file\n");}
        write_output_raw(   temp,
                            temp_filename,
                            size,
                            proc_size,
                            new_proc_size,
                            starts,
                            comm_cart,
                            dims,
                            rank,
                            n_proc,
                            cart_coords,
                            MPI_DOUBLE,
                            false,
                            dx);
    }

    if (write_output[1] == 1){
        if (rank == 0){printf("Writing domain decomposition to file\n");}
        write_domain_decomposition( domain_decomposition,
                                    domd_filename,
                                    size,
                                    proc_size,
                                    new_proc_size,
                                    starts,
                                    comm_cart,
                                    dims,
                                    rank,
                                    n_proc,
                                    cart_coords,
                                    MPI_INT);
    }

    MPI_Finalize();
}
