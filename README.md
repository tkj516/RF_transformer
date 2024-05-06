
# RF Transformer

## Pre-process and create the interference datasets

Download the inteference frames from [here](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0)

```
cd utils
python preprocess_ICASSP_dataset.py \
--root_dir=<PATH_TO_SAVE_DATASET> \
--interference_sig_type=<SIGNAL_NAME> \
--dataset_path=<PATH_TO_INTERFERENCE_FRAMES>
```

For example, to create a dataset of CommSignal2 frames,
```
python preprocess_ICASSP_dataset.py \
--root_dir=../dataset/TestSet1 \
--interference_sig_type=CommSignal2 \
--dataset_path=../../dataset/interferenceset_frame/CommSignal2_raw_data.h5
```

## Training

Run the training script by running the following command.  You can create new config
files or new training scripts and use the same infrastructure to train a new model.

```
python main.py --trainer=transformer --config=qpsk_ofdm_qpsk_simple_config.yaml
```
