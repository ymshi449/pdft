FC=gfortran
FFLAGS=-finit-local-zero -O3 -ffixed-line-length-132 -Wall -g -fbounds-check -fbacktrace
F90FLAGS=-finit-local-zero -O0 -ffree-line-length-none -x f95-cpp-input -Wall -g -fbounds-check -fbacktrace
# I use O0 above just to remove warning that __result_func’ may be used uninitialized in function func
QUENCH  = -L$(HOME)/SimulatedAnnealing/quench_anneal/lib -lquench -lquench_seq
LINPACK = -L$(HOME)/lib2/linpack -llinpack
BLAS = -L$(HOME)/lib2/blas -lblas
#LIBS = $(LINPACK) $(QUENCH)
LIBS = $(QUENCH) $(LINPACK) $(BLAS)

.SUFFIXES:
.SUFFIXES: .f90 .f95 .o .f .c

.f90.o:
	$(FC) $(F90FLAGS) -o $@ -c $<

# OBJS = errors.o covar.o
OBJS = covar.o

all: covar

covar: $(OBJS)
	$(FC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@
