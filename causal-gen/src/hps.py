import argparse

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


mimic = Hparams()
# training
mimic.epochs = 1000
mimic.bs = 16
# mimic.lr = 0.001
mimic.lr_warmup_steps = 100
mimic.wd = 0.001
mimic.betas = [0.9,0.9]
mimic.ema_rate = 0.999
mimic.input_res = 256
mimic.input_channels = 1
mimic.grad_clip = 100
mimic.grad_skip = 1000
mimic.accu_steps = 1
# mimic.beta = 9.0
mimic.beta_warmup_steps = 0
mimic.kl_free_bits = 0.0
mimic.eval_freq = 1
mimic.viz_freq = 3000 # approx after 1 epoch for mimic with bs 1
mimic.scale_range = 0.1
mimic.rotation_degree = 10.0

# model
mimic.enc_arch = "256b1d2,128b3d2,64b7d2,32b11d2,16b7d2,8b3d8,1b2"
mimic.dec_arch = "1b2,8b4,16b8,32b12,64b8,128b4,256b2"
mimic.widths = [32, 64, 96, 128, 160, 192, 512]
mimic.bottleneck = 4
mimic.z_dim = [48, 30, 24, 18, 12, 6, 1]
mimic.z_max_res = 128
mimic.bias_max_res = 64
mimic.x_like = "fixed_dgauss"
# mimic.std_init = 1e-2
mimic.parents_x = ["sex", "age", "view", "finding", "race",]
mimic.context_dim = 14 # 2 (sex:male/female) + 1 (age:0-100) + 3 (race:white/black/asian) + 2+1 (view:PA/AP) + 4+1 (finiding:NF/PE/CM/PE&CM) 
mimic.dummy_var = ["view", "finding"]
mimic.embd_dim = 32
mimic.p_dropout = 0.1

#dataloader
mimic.rca_threshold = 0.8
mimic.test_csv = "test_pe_cm.csv"
mimic.valid_csv = "valid_pe_cm.csv"
mimic.csvpath = "/vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/"
mimic.inputpath = "/vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_files_preprocessed/"
mimic.labelpath = "/vol/biomedic3/rmehta3/CheXMask/datasets/chest_xray/mimic-chexmask-jpg-256x256/CheXMask_segmentation_preprocessed/"


HPARAMS_REGISTRY["mimic"] = mimic


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="mimic")
    parser.add_argument("--dataset", help="dataset name.", type=str, default="mimic")
    parser.add_argument("--resume", help="Path to load checkpoint.", type=str, default="")
    parser.add_argument("--seed", help="Set random seed.", type=int, default=42)
    parser.add_argument("--deterministic",help="Toggle cudNN determinism.",action="store_true",default=False,)
    #dataloader
    parser.add_argument("--inputpath", help="Data directory to load form.", type=str, default="")
    parser.add_argument("--labelpath", help="Label (segmentation) directory to load form.", type=str, default="")
    parser.add_argument("--csvpath", help="path to directory where csv files are stored.", type=str, default="")
    parser.add_argument("--train_csv", help="csv file name for train set.", type=str, default="train_pe_cm.csv")
    parser.add_argument("--test_csv", help="csv file name for test set.", type=str, default="test_pe_cm.csv")
    parser.add_argument("--valid_csv", help="csv file name for valid set.", type=str, default="valid_pe_cm.csv")
    parser.add_argument("--rca_threshold", help="rca threshold for mimic dataframe", type=float, default=0.8)
    parser.add_argument("--weightedsampler", help="if True, weightedRandomSampler is used.", action="store_true", default=False)
    # training
    parser.add_argument("--epochs", help="Training epochs.", type=int, default=5000)
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", help="lr warmup steps.", type=int, default=100)
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.01)
    parser.add_argument("--betas", help="Adam beta parameters.", nargs="+", type=float, default=[0.9, 0.9],)
    parser.add_argument("--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999)
    parser.add_argument("--input_res", help="Input image crop resolution.", type=int, default=64)
    parser.add_argument("--input_channels", help="Input image num channels.", type=int, default=1)
    parser.add_argument("--grad_clip", help="Gradient clipping value.", type=float, default=350)
    parser.add_argument("--grad_skip", help="Skip update grad norm threshold.", type=float, default=500)
    parser.add_argument("--accu_steps", help="Gradient accumulation steps.", type=int, default=1)
    parser.add_argument("--beta", help="Max KL beta penalty weight.", type=float, default=3.0)
    parser.add_argument("--beta_warmup_steps", help="KL beta penalty warmup steps.", type=int, default=0)
    parser.add_argument("--kl_free_bits", help="KL min free bits constraint.", type=float, default=0.0)
    parser.add_argument("--viz_freq", help="Steps per visualisation.", type=int, default=10000)
    parser.add_argument("--eval_freq", help="Train epochs per validation.", type=int, default=5)
    parser.add_argument("--scale_range", help="Augmentation: Scale in the range of 1+-scale_range.", type=float, default=0.1)
    parser.add_argument("--rotation_degree", help="Augmentation: Rotation (degree) in the range of +-rotation_degree.", type=float, default=10.0)
    # model
    parser.add_argument("--enc_arch", help="Encoder architecture config.", type=str,default="64b1d2,32b1d2,16b1d2,8b1d8,1b2",)
    parser.add_argument("--dec_arch", help="Decoder architecture config.", type=str, default="1b2,8b2,16b2,32b2,64b2",)
    parser.add_argument("--widths", help="Number of channels.", nargs="+", type=int, default=[16, 32, 48, 64, 128],)
    parser.add_argument("--bottleneck", help="Bottleneck width factor.", type=int, default=4)
    parser.add_argument("--z_dim", help="Number of latent channel dims.", nargs="+", type=int, default=[48, 30, 24, 18, 12, 6, 1])
    parser.add_argument("--z_max_res", help="Max resolution of stochastic z layers.", type=int, default=192,)
    parser.add_argument("--bias_max_res", help="Learned bias param max resolution.", type=int, default=64,)
    parser.add_argument("--x_like", help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.", type=str, default="diag_dgauss",)
    parser.add_argument("--std_init", help="Initial std for x scale. 0 is random.", type=float, default=0.0,)
    parser.add_argument("--parents_x", help="Parents of x to condition on.", nargs="+",default=["view", "age", "race", "sex", "finding"],)
    parser.add_argument("--add_dummy_dim", help="add dummy dimension for softmax_centered trick in PGM", action="store_true", default=False)
    parser.add_argument("--dummy_var", help="Variable which requires added dummy dimensions in PGM.", nargs="+", default=["view", "finding"])
    parser.add_argument("--context_dim", help="Num context variables conditioned on.", type=int, default=4,)
    parser.add_argument("--embd_dim", help="Embedding dim", type=int, default=32,)
    parser.add_argument("--p_dropout", help="Block dropout", type=float, default=0.1,)
    parser.add_argument("--concat_pa", help="Whether to concatenate parents_x.", action="store_true", default=False,)
    parser.add_argument("--cond_prior", help="Use a conditional prior.", action="store_true", default=False,)
    parser.add_argument("--q_correction", help="Use posterior correction.", action="store_true", default=False,)
    parser.add_argument("--cond_drop", help="Use counterfactual dropout", action="store_true", default=False,)
    return parser
