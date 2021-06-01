#!/bin/bash
#
#SBATCH --job-name=spec_test1
#SBATCH --output=/home/mvasist/scripts/assessment/alan-log/slurm-%j.out
#SBATCH --time="2-00:00:00" 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G

# print the name of the node on which the job is running:
echo "Job running on $SLURMD_NODENAME"

### Copy the data to the Alan machine: ###

HOME_DIR="/home/mvasist/scripts/assessment/datasets/"
NODE_DIR="/scratch/users/mvasist/assessment/"
mkdir -p $NODE_DIR # Create the directory if it doesn't already exist

# HOME_DIR="/home/malavika/Documents/Research/assessment/"
# NODE_DIR="datasets/"

FILE_NAME="4params_train_bce.h5"

FILE_LOC_NODE="$NODE_DIR$FILE_NAME"


if ! [ -f "$FILE_LOC_NODE" ]; then
    echo "File "$FILE_LOC_NODE" does not already exists"
    SECONDS=0
    cp -r "$HOME_DIR/$FILE_NAME" "$NODE_DIR"
    duration=$SECONDS
    echo "File copied in $(($duration / 60))m$(($duration % 60))s"
    else
        echo "File "$FILE_LOC_NODE" already exists"
fi 
      
### Activate the virtual environment: ###
source ~/miniconda3/etc/profile.d/conda.sh
conda activate petitRT

### Go to the proper directory: ###
cd /home/mvasist/scripts/assessment/training/

### Run the python script: ###
echo "Running the training script"
python training.py -floc $FILE_LOC_NODE -dla "mlp" -dsiz 10000 -opt "Adam" -bs 128 -lr '1e-4' -nep 5 -splt "90" -met 'bce' -nparam 4


# #for terminal in local system
# python training.py -floc '/home/malavika/Documents/Research/assessment/datasets/4params_train.h5' -dla "mlp" -dsiz 117882 -opt "Adam" -bs 128 -lr '1e-4' -nep 10 -splt "90" -met 'bce' -nparam 4


#mkdir -p /scratch/users/mvasist/assessment/
#cp -r /home/mvasist/scripts/assessment/datasets/4params_train_bce.h5 /scratch/users/mvasist/assessment/.
#python /home/mvasist/scripts/assessment/training/training.py -floc /scratch/users/mvasist/assessment/4params_train_bce.h5 -dla "mlp" -dsiz 117882 -opt "Adam" -bs 128 -lr '1e-4' -nep 1 -splt "90" -met 'bce' -nparam 4
#sbatch /home/mvasist/scripts/assessment/training/submit_training.sh


##SBATCH --exclude=alan-compute-[06-09]
##SBATCH --nodelist=alan-compute-02