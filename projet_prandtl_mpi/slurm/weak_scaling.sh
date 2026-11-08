#!/bin/bash
#SBATCH --job-name=prandtl_weak
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=logs/weak_%j.log
# Passage a l'echelle FAIBLE : le cout d'un produit matrice-vecteur est en O(N^2/P) ;
# pour garder la charge de calcul par coeur constante on prend N = N0 * sqrt(P) (et non N ~ P).
# Usage : sbatch slurm/weak_scaling.sh
#         N0=4096 sbatch slurm/weak_scaling.sh
set -eu
module load gcc openmpi
mkdir -p logs data
make -s

N0=${N0:-4096}
NREP=${NREP:-5}
PLIST=${PLIST:-"1 2 4 8 16"}
LAUNCH=${LAUNCH:-"srun -n"}
OUT="data/weak_N0_${N0}.csv"
rm -f "$OUT"

for P in $PLIST; do
  N=$(awk -v n0="$N0" -v p="$P" 'BEGIN{n=int(n0*sqrt(p)+0.5); if (n%2) n++; print n}')
  for OV in 0 1; do
    echo "--- P=$P  N=$N  recouvrement=$OV"
    $LAUNCH "$P" ./bin/prandtl n="$N" rep="$NREP" recouvrement="$OV" sortie="$OUT" | grep -E "^CSV|Gradient"
  done
done
echo "[INFO] Campagne faible terminee : $OUT"
