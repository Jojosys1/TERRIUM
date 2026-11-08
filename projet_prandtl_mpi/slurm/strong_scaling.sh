#!/bin/bash
#SBATCH --job-name=prandtl_strong
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/strong_%j.log
# Passage a l'echelle FORTE : N fixe, P = 1, 2, 4, 8, 16, avec (1) et sans (0) recouvrement.
# Usage : sbatch slurm/strong_scaling.sh
#         N=16384 NREP=5 sbatch slurm/strong_scaling.sh
set -eu
module load gcc openmpi
mkdir -p logs data
make -s

N=${N:-8192}
NREP=${NREP:-5}
PLIST=${PLIST:-"1 2 4 8 16"}
LAUNCH=${LAUNCH:-"srun -n"}            # sinon : LAUNCH="mpirun -np"
OUT="data/strong_N${N}.csv"
rm -f "$OUT"

for P in $PLIST; do
  for OV in 0 1; do
    echo "--- P=$P  N=$N  recouvrement=$OV"
    $LAUNCH "$P" ./bin/prandtl n="$N" rep="$NREP" recouvrement="$OV" sortie="$OUT" | grep -E "^CSV|Gradient"
  done
done
echo "[INFO] Campagne forte terminee : $OUT"
