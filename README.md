
# RF Transformer

This repository contains code for various experiments centered around using transformers for generative modeling and source separation.

## Installing Anaconda Environment

    conda env create -f environment.yml

## Repository Structure

Broadly, an experiment is defined by a configuration file that is called within a general training script.  In order to make the configurations modular and flexible, we use the `ml_collections` package.  The script `main.py` is the main entrypoint to any experiment.  It uses `ml_collections` along with `abseil` to enable a command line-like interface and wrap all the all code related to setting up the hardware configuration for training, e.g., distributed training.

As an example, to launch a transformer experiment that uses a decoder-only transformer for source separation of a QPSK + CommSignal2 mixture with SOI synchronization can be launched as follows.
```
python main.py --trainer=transformer_decoder --config=configs/qpsk_commsignal2_unsynchronized_decoder.py
```
The `--trainer` flag denotes which training script/experiment is intended and the `--config` flag specifies the location of the configuration file.

The overall repository for a single experiment is structured as follows:

    .
    ├── main.py                  
    ├── train_<EXPERIMENT>.py
    ├── models
    ├── configs
    ├── rfcutils2
    ├── eval
    ├── utils
    ├── dataset
    ├── checkpoints
    └── README.md

## Models and Registers

Each time you define a new model/callable, it is recommended that you specify default parameters or explicity mark which arguments are required keyword arguments. Within each training script, various registers are defined.  The goal of a register to allow for easy testing of different architectures, loss functions, optimizers etc. 

Within the configuration file, you can define specific architectural configurations that you would like to experiment with. Consider the following snippet,

    config.model_config = [
        "Transformer",
        ml_collections.ConfigDict(
            dict(
                input_dim=320,
                output_dim=256,
                n_encoder_layers=12,
                n_decoder_layers=12,
                embed_dim=768,
                n_head=12,
                bias=False,
                dropout=0.0,
                block_size=320,
                causal_encoder=False,
                causal_decoder=True,
                max_seq_len=320,
            )
        ),
    ]

Here the model is defined as a `Transformer` with certain specific attributes.  Refer to the actual definition of the model in `models/transformer_decoder.py`:

    class Transformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_decoder_layers: int,
            embed_dim: int,
            n_head: int,
            bias: bool,
            dropout: float,
            block_size: int,
            causal_decoder: bool,
            max_seq_len: int,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.embed_dim = embed_dim
            self.n_head = n_head
            self.bias = bias
            self.dropout = dropout
            self.block_size = block_size
            self.max_seq_len = max_seq_len
            self.n_decoder_layers = n_decoder_layers

            self.input_projection = nn.Linear(input_dim, embed_dim)

            self.decoder = Decoder(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=n_decoder_layers,
                embed_dim=embed_dim,
                n_head=n_head,
                bias=bias,
                dropout=dropout,
                block_size=block_size,
                causal=causal_decoder,
            )
            self.freqs_cis = precompute_freqs_cis(embed_dim // n_head, max_seq_len)

            # Init all weights
            self.apply(self._init_weights)
            # Apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith("output_proj.weight"):
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=0.02 / math.sqrt(n_decoder_layers),
                    )

        @property
        def num_params(self) -> int:
            n_params = sum(p.numel() for p in self.parameters())
            return n_params

        def _init_weights(self, module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        def embed_patch(self, input: torch.Tensor) -> torch.Tensor:
            return self.input_projection(input)

        def forward(
            self, input: torch.Tensor, start_pos: int = 0
        ) -> torch.Tensor:
            _, t, _ = input.shape
            assert t <= self.block_size, (
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.block_size}"
            )

            freqs_cis = self.freqs_cis.to(input.device)
            freqs_cis = freqs_cis[start_pos : start_pos + t]

            # Get the decoded features
            pred = self.decoder(input, freqs_cis)

            return pred

        def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
            """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
            # first estimate the number of flops we do per iteration.
            # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
            N = self.get_num_params()
            L, H, Q, T = (
                self.n_decoder_layers,
                self.n_head,
                self.embed_dim // self.n_head,
                self.block_size,
            )
            flops_per_token = 6 * N + 12 * L * H * Q * T
            flops_per_fwdbwd = flops_per_token * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
            # express our flops throughput as ratio of A100 bfloat16 peak flops
            flops_achieved = flops_per_iter * (1.0 / dt)  # per second
            flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
            mfu = flops_achieved / flops_promised
            return mfu

        @torch.no_grad()
        def generate(
            self,
            input: torch.Tensor,
            start_pos: int = 0,
        ) -> torch.Tensor:
            T = input.shape[1]

            freqs_cis = self.freqs_cis.to(input.device)
            freqs_cis = freqs_cis[start_pos : start_pos + T]

            recons = self.decoder(input, freqs_cis)

            return recons

Note that in the config, not all keywords are specified.  This is where the register comes in handy.  In the backend, once a model is registered, the script automatically parses the model constructor and creates an internal dataclass that is aware of the default argument values.  Therefore, no explicit dataclasses or additional code needs to written everytime a new model is defined. Just register the model and create a configuration file to test our your new architecture!

An example register to test our different optimizers is:

    OPTIMIZER_REGISTER = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
    }
    optimizer_builder = ClassBuilder(OPTIMIZER_REGISTER)

## Create the synthetic SOI datasets

A synthetic 16 x oversampled QPSK dataset with samples of length 40976 (2561 symbols) 
can be created by running the command below.  Note that the synchronous dataset has 
samples of length 40960 but since we want random shifts during training we generate a 
QPSK signal withan extra symbol (one symbol = 16 samples)

```
python prepare_synthetic_datasets.py \
--num_samples=200000 \
--signal_length=2561 \
--root_dir=qpsk \
--signal_name=qpsk \
qpsk
```

As another example a synthetic OFDM (QPSK) signal with 512 symbols 
(one symbol = 80 samples) can be generated as 

```
python prepare_synthetic_datasets.py \
--num_samples=200000 \
--signal_length=512 \
--root_dir=ofdm \
--signal_name=ofdm \
ofdm
```

If you want to change the underlying constellation you can modify the following line
in `rfcutils/ofdm_helper_fn.py`
 
```
# K-QAM constellation
NUM_BITS_PER_SYMBOL = <NUMBER_BITS_PER_SYMBOL>
```

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
