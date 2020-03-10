'''This is a function to run many different instances on the cluster
'''
import numpy as np
import os
import shutil
import argparse

measurement_noises = [0.005, 0.01, 0.05, 0.1, 0.15]
use_mm = [0,1]

my_str = '''
#!/bin/bash
#BSUB -J pylab
#BSUB -o {3}.out
#BSUB -e {3}.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q big-multi
#BSUB -n 8
#BSUB -M 10000
#BSUB -R rusage[mem=10000]

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here
# Load module
module load anaconda/default
source activate dispatcher

cd /PHShome/jjd65/prod_of_experts/
python3 ./main.py -MM {0} -a {1} -o {2} -gstepps {3} -useF1 {4}
'''

# Make the directories to store the information

#try:
#    os.mkdir(basepath)
#except OSError:
    # Remove the diinconsistent use of tabs and spaces in indentationrectory and then make one
#    shutil.rmtree(basepath)
#    os.mkdir(basepath)

options = {'cooperation3','competing2','competing3a','competing3b'}
useF1 = {True, False}
basepath = 'outdir'

for m in use_mm:
    for uf1 in useF1:
        for opt in options:
            outdir = 'outdir_new_opt_' + opt + '_MM' + str(m) + '_useF1_' + str(uf1)
            print(outdir)

            fname = outdir + '.lsf'

            f = open(fname,'w')
            f.write(my_str.format(m,opt,outdir,1000,uf1))
            f.close()
            os.system('bsub < {}'.format(fname))
