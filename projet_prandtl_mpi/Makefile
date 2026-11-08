# Projet 2 MSN — Equation de Prandtl en parallele (Fortran 2008 + mpi_f08)
FC       = mpif90
# -march=native : compiler sur une machine du type des noeuds de calcul (sinon retirer l'option)
FFLAGS   ?= -O3 -march=native
STRICT   := -std=f2008 -Wall -Wextra -pedantic
DEBUG    := -O0 -g -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace

SRC := precision_mod parametres_mod metrologie_mod decomposition_mod aile_mod solveur_mod main
OBJ := $(addprefix build/,$(addsuffix .o,$(SRC)))

all: bin/prandtl

bin/prandtl: $(OBJ)
	@mkdir -p bin
	$(FC) $(FFLAGS) $(STRICT) $^ -o $@

build/%.o: src/%.f90
	@mkdir -p build
	$(FC) $(FFLAGS) $(STRICT) -J build -c $< -o $@

# dependances entre modules
build/parametres_mod.o:    build/precision_mod.o
build/metrologie_mod.o:    build/precision_mod.o
build/decomposition_mod.o:
build/aile_mod.o:          build/parametres_mod.o
build/solveur_mod.o:       build/parametres_mod.o build/metrologie_mod.o build/decomposition_mod.o build/aile_mod.o
build/main.o:              build/solveur_mod.o

# version de mise au point : controles d'indices, exceptions flottantes
debug:
	@mkdir -p build_debug bin
	$(FC) $(DEBUG) $(STRICT) -J build_debug $(addprefix src/,$(addsuffix .f90,$(SRC))) -o bin/prandtl_debug

# validation automatique (theorie, independance vis-a-vis de P et du recouvrement)
test: all
	bash tests/valider.sh

clean:
	rm -rf build build_debug bin resultats.csv profil_gamma.csv

.PHONY: all debug test clean
