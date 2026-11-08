#!/bin/bash
# Validation automatique du solveur de Prandtl.
#   1. aile elliptique : CL et efficacite d'envergure comparees a la theorie ;
#   2. resultats independants du nombre de processus P et du mode de communication ;
#   3. aile trapezoidale vrillee comparee a la solution de Glauert (si python3 + numpy).
# Usage : bash tests/valider.sh        (LAUNCH="srun -n" pour un lancement SLURM)
cd "$(dirname "$0")/.." || exit 1
LAUNCH=${LAUNCH:-mpirun -np}
BIN=./bin/prandtl
TMP=$(mktemp -d)
echec=0

verifier() {   # verifier "description" <0|1>
  if [ "$2" = "1" ]; then echo "  [OK]    $1"; else echo "  [ECHEC] $1"; echec=1; fi
}
dans() { awk -v v="$1" -v lo="$2" -v hi="$3" 'BEGIN{print (v>=lo && v<=hi) ? 1 : 0}'; }

echo "1) Aile elliptique (N = 2000, P = 3)"
sortie=$($LAUNCH 3 $BIN n=2000 elliptique=1 vrillage=0 rep=1 sortie="$TMP/e.csv" </dev/null)
ecart=$(echo "$sortie" | awk '/ecart CL/ {print $(NF-1)}')
effic=$(echo "$sortie" | awk '/Efficacite/ {print $NF}')
verifier "ecart de CL a la theorie = $ecart % (attendu < 0.05 %)" "$(dans "$ecart" -0.05 0.05)"
verifier "efficacite d'envergure e = $effic (attendu 1.000 a 1e-3)" "$(dans "$effic" 0.999 1.001)"

echo "2) Independance vis-a-vis de P et du recouvrement (N = 1000, cas par defaut)"
for P in 1 2 3 5; do
  for OV in 0 1; do
    $LAUNCH $P $BIN n=1000 rep=1 recouvrement=$OV sortie="$TMP/p.csv" </dev/null > /dev/null || echec=1
  done
done
ecarts=$(awk -F, 'NR==2 {cl0=$12; cd0=$13} NR>1 {d1=($12-cl0)/cl0; d2=($13-cd0)/cd0;
  if (d1<0) d1=-d1; if (d2<0) d2=-d2; if (d1>m1) m1=d1; if (d2>m2) m2=d2; if ($14>mr) mr=$14}
  END {printf "%.2e %.2e %.2e", m1, m2, mr}' "$TMP/p.csv")
set -- $ecarts
verifier "ecart max de CL entre configurations = $1 (attendu < 1e-7)" "$(dans "$1" 0 1e-7)"
verifier "ecart max de CDi entre configurations = $2 (attendu < 1e-7)" "$(dans "$2" 0 1e-7)"
verifier "residu vrai max = $3 (attendu < 1e-9)" "$(dans "$3" 0 1e-9)"

echo "3) Aile trapezoidale vrillee et cambree vs Glauert (N = 4000, P = 3)"
if python3 -c "import numpy" 2>/dev/null; then
  ref=$(python3 tests/glauert.py 3 0.5 0.2 2 -3 -4.15)
  sortie=$($LAUNCH 3 $BIN n=4000 envergure=3 croot=0.5 ctip=0.2 incidence=2 vrillage=-3 alpha_l0=-4.15 \
           rep=1 sortie="$TMP/g.csv" </dev/null)
  cl=$(echo "$sortie" | awk '/CL  =/ {print $3}'); cdi=$(echo "$sortie" | awk '/CL  =/ {print $6}')
  ecl=$(awk -v a="$cl" -v b="$(echo $ref | cut -d' ' -f1)" 'BEGIN{printf "%.4f", 100*(a/b-1)}')
  ecd=$(awk -v a="$cdi" -v b="$(echo $ref | cut -d' ' -f2)" 'BEGIN{printf "%.4f", 100*(a/b-1)}')
  verifier "ecart de CL vs Glauert = $ecl % (attendu < 0.1 %)" "$(dans "$ecl" -0.1 0.1)"
  verifier "ecart de CDi vs Glauert = $ecd % (attendu < 0.2 %)" "$(dans "$ecd" -0.2 0.2)"
else
  echo "  [IGNORE] python3 + numpy absents"
fi

rm -rf "$TMP"
if [ $echec -eq 0 ]; then echo "Validation reussie."; else echo "Validation ECHOUEE."; fi
exit $echec
