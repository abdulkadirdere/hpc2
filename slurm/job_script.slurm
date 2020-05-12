#!/bin/bash
# These lines are for slurm
# specify a partition by its name in which to run your job . In
# this case , the name of the partition is batch
#SBATCH -p batch
# specify number of nodes you are requesting to run your job
#SBATCH -N 2
# specify number of cores or tasks
#SBATCH -n 8
# specify memory pool for all cores in MBs. In this case 10 MB is requested
##SBATCH --mem 10
# specify the wall clock time limit for the job. In this case, it is 5 mins
#SBATCH -t 00:10:00
# specify the job name
#SBATCH -J test-job
# specify the filename to be used for writing output . You need
# to put the correct path and the output file name accordingly
#SBATCH -o /home-mscluster/adere/HPC/slurm.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/<username>/slurm.%N.%j.err
# The following are shell commands -- which will be executed
# according to the settings specified above