#!/bin/bash

MIXTURES=('CommSignal2' 'CommSignal3' 'CommSignal5G1' 'EMISignal1')
TESTSET=('TestSet2Mixture')

for mixture in ${MIXTURES[@]}; do
    for testset in ${TESTSET[@]}; do
        python evaluate_synchronized.py \
        --checkpoint_dir=$1 \
        --interference_sig_type=$mixture \
        --testset_identifier=$testset \
        --id_string=$2

        python process_results.py \
        --interference_sig_type=$mixture \
        --testset_identifier=$testset \
        --id_string=$2
    done
done