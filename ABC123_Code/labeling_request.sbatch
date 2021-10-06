#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl20
#SBATCH --time=01:00:00
#SBATCH --output=labeling_%j.out
#SBATCH --error=labeling_%j.err
#SBATCH --exclusive
#SBATCH --requeue 

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 


mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/DL21SP20/code_20

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python labeling_request.py --checkpoint-dir ./checkpoints \
--checkpoint-file finetuned_barlowtwins_resnet50_epochs100.pth
"
