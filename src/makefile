# -----------------------------------------------------------------------
# HEATMAPS — Finite Difference Heat Conduction Solver
# -----------------------------------------------------------------------
# Override the compiler and flags on the command line if needed, e.g.:
#   make MPICXX=mpic++ CXXFLAGS="-O2 -std=c++17"
# -----------------------------------------------------------------------

MPICXX   ?= mpiCC
CXXFLAGS  = -O3 -std=c++14 -Wall
LDFLAGS   = -lm
TARGET    = ../bin/HEATMAPS

SRCS = read_problem_params.cc \
       geometry.cc            \
       output.cc              \
       parallelization.cc     \
       initialize.cc          \
       calc_heat.cc           \
       solver.cc              \
       boundary_conditions.cc \
       evaluation.cc          \
       main_heat_solver.cc

OBJS = $(SRCS:.cc=.o)

# -----------------------------------------------------------------------
.PHONY: all clean debug

all: $(TARGET)

debug: CXXFLAGS = -O0 -g -std=c++14 -Wall
debug: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile (pattern rule covers all translation units)
%.o: %.cc
	$(MPICXX) $(CXXFLAGS) -c -o $@ $<

# -----------------------------------------------------------------------
# Header dependencies (keep in sync with #includes in each .cc file)
# -----------------------------------------------------------------------
read_problem_params.o:  read_problem_params.cc read_problem_params.h

geometry.o:             geometry.cc geometry.h \
                        constants.h parallelization.h

output.o:               output.cc output.h \
                        constants.h

parallelization.o:      parallelization.cc parallelization.h \
                        constants.h

initialize.o:           initialize.cc initialize.h \
                        constants.h parallelization.h

calc_heat.o:            calc_heat.cc calc_heat.h \
                        constants.h parallelization.h solver.h \
                        boundary_conditions.h \
                        evaluation.h output.h

solver.o:               solver.cc solver.h \
                        constants.h

boundary_conditions.o:  boundary_conditions.cc boundary_conditions.h \
                        constants.h

evaluation.o:           evaluation.cc evaluation.h \
                        constants.h

main_heat_solver.o:     main_heat_solver.cc \
                        constants.h read_problem_params.h geometry.h \
                        output.h calc_heat.h parallelization.h initialize.h

# -----------------------------------------------------------------------
clean:
	rm -f $(OBJS) $(TARGET)
