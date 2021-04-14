#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl20
#SBATCH --time=0:10:00
#SBATCH --output=train_logs/simsiam_%j.out
#SBATCH --error=train_logs/simsiam_%j.err
#SBATCH --exclusive
#SBATCH --requeue 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/DL21SP20

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python /SimSiam/simsiam_train.py --checkpoint-dir $HOME/checkpoints/simsiam --model-name simsiam --epochs=1
"
#python eval.py --checkpoint-path $HOME/checkpoints/models/baseline.pth
