#!/bin/bash 
cd $ZOE_WORKSPACE/cgp_mnist 
echo "Starting batch job" 
PYTHONPATH=. python ./dgp_rff_mnist.py --seed=12345 --train_size=1000 --kernel_type=RBF --optimizer=adam --nl=1 --learning_rate=0.001 --n_rff=200 --df=200 --batch_size=1000 --mc_train=1 --mc_test=20 --n_iterations=100001 --display_step=1000 --less_prints=True --duration=1200000 --theta_fixed=4000 --feed_forward=False --learn_Omega=optim > ../mnist_1000train_cgp_1nl_200rff_200df_1mctrain_20mctest_4000tf_0ff_optimOmega.txt
