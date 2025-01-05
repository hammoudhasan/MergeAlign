# 100,200, 500 samples with 100 steps
#sbatch script_multinode_restarts.sh 0.3 1 100 100
#sbatch script_multinode_restarts.sh 0.3 1 200 100
#sbatch script_multinode_restarts.sh 0.3 1 500 100

# 1000 samples with different steps
#sbatch script_multinode_restarts.sh 0.3 1 1000 10
#sbatch script_multinode_restarts.sh 0.3 1 1000 50
#sbatch script_multinode_restarts.sh 0.3 1 1000 200
#sbatch script_multinode_restarts.sh 0.3 1 1000 300

#sbatch script_multinode_restarts.sh 0.0 1 1000 50

#sbatch script_multinode_restarts.sh 0.3 4 1000 100
#sbatch script_multinode_restarts.sh 0.3 32 1000 100



sbatch script_multinode_restarts.sh 0.0 1.0 1 1000 100 slerp
sbatch script_multinode_restarts.sh 0.0 1.0 8 1000 100 slerp
sbatch script_multinode_restarts.sh 0.0 1.0 16 1000 100 slerp

sbatch script_multinode_restarts.sh 0.0 1.0 1 1000 100 ties
sbatch script_multinode_restarts.sh 0.0 1.0 8 1000 100 ties
sbatch script_multinode_restarts.sh 0.0 1.0 16 1000 100 ties

sbatch script_multinode_restarts.sh 0.0 1.0 1 1000 100 dare_ties
sbatch script_multinode_restarts.sh 0.0 1.0 8 1000 100 dare_ties
sbatch script_multinode_restarts.sh 0.0 1.0 16 1000 100 dare_ties


