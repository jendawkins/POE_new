'''This is a function to run many different instances on the cluster
'''
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-otus', '-o', type=int,
    help='Number of OTUs', dest='n_otus', default=150)
parser.add_argument('--n-data-seeds', '-d', type=int,
    help='Number of data seeds for each noise level', 
    dest='n_data_seeds', default=5)
parser.add_argument('--n-init-seeds', '-i', type=int,
    help='Number of initialization seeds for each data seed', 
    dest='n_init_seeds', default=1)
parser.add_argument('--percent-change-clustering', '-pcc', type=float,
        help='Percent of OTUs to update during clustering every time it runs',
        default=1.0, dest='percent_change_clustering')
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=4000)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=2000)
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
args = parser.parse_args()

measurement_noises = [0.01, 0.05, 0.1, 0.15]
process_variances = [0.01]
data_seed = np.arange(0,args.n_data_seeds)
init_seed = np.arange(0,args.n_init_seeds)
if args.basepath is None:
    basepath = 'OTUs{}/'.format(args.n_otus)
else:
    basepath = args.basepath

my_str = '''
#!/bin/bash
#BSUB -J pylab
#BSUB -o {5}_{3}.out
#BSUB -e {5}_{3}.err

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

cd /PHShome/dek15/perturbation_study/
python main_ICML.py -m {0} -p {1} -d {2} -i {3} -b {4} -n {6} -q {0} -nb {7} -ns {8} -pcc {9}
'''

# Make the directories to store the information

try:
    os.mkdir(basepath)
except OSError:
    # Remove the directory and then make one
    shutil.rmtree(basepath)
    os.mkdir(basepath)


for mn in measurement_noises:
    mnfolder = basepath + 'measurement_noise_{}/'.format(int(100*mn))
    os.mkdir(mnfolder)

    for p in process_variances:
        pfolder = mnfolder + 'process_variance_{}/'.format(int(100*p))
        os.mkdir(pfolder)

        for ds in data_seed:
            dsfolder = pfolder + 'data_seed_{}/'.format(ds)
            os.mkdir(dsfolder)
            logname = dsfolder + 'logs/'
            os.mkdir(logname)

            for i in init_seed:

                fname = dsfolder + 'init_seed_{}.lsf'.format(i)
                

                f = open(fname, 'w')
                f.write(my_str.format(mn,p,ds,i, dsfolder, logname,args.n_otus, 
                    args.burnin, args.n_samples, args.percent_change_clustering))
                f.close()

                import pdb; pdb.set_trace()
                os.system('bsub < {}'.format(fname))
