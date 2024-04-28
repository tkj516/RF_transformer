!/bin/bash

python evaluate_unsynchronized.py \
--soi_root_dir=dataset/qpsk/qpsk_50_2561 \
--interference_root_dir=dataset/TestSet1/CommSignal2 \
--checkpoint_dir=checkpoints/qpsk_commsignal2_unsynchronized/weights.pt