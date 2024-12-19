#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ece.canli@uni-ulm.de
#SBATCH --output=testout.log
#SBATCH --error=testerr.log
echo "Activating venv environment"
source activate ./venv/bin/activate
source ./secrets.sh

python ./src/eval_preranked.py "$@"
