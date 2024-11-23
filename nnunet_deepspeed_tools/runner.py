from os.path import join
from typing import Union

import torch

from argparse import Namespace
from batchgenerators.utilities.file_and_folder_operations import load_json
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
from pyspark.sql import SparkSession
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunet_deepspeed_tools import DeepSpeedTrainer

from .configs import *


def get_command_line_arguments() -> Namespace:
    import argparse
    import deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_or_id", type=str, help="Dataset name or ID to train with")
    parser.add_argument("--configuration", type=str, help="Configuration that should be trained")
    parser.add_argument("--fold", type=str,
                        help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.")
    parser.add_argument("--pretrained_weights", type=str, required=False, default=None,
                        help="[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only "
                             "be used when actually training. Beta. Use with caution.")
    parser.add_argument("--num_gpus", type=int, default=1, required=False,
                        help="Specify the number of GPUs to use for training")
    parser.add_argument("--num_nodes", type=int, default=1, required=False,
                        help="Specify the number of worker nodes to use for training")
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument("--npz", action="store_true", required=False,
                        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
                             "segmentations). Needed for finding the best ensemble.")
    parser.add_argument("--c", action="store_true", required=False,
                        help="[OPTIONAL] Continue training from latest checkpoint")
    parser.add_argument("--val", action="store_true", required=False,
                        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.")
    parser.add_argument("--val_best", action="store_true", required=False,
                        help="[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead "
                             "of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! "
                             "WARNING: This will use the same \"validation\" folder as the regular validation "
                             "with no way of distinguishing the two!")
    parser.add_argument("--disable_checkpointing", action="store_true", required=False,
                        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
                             "you dont want to flood your hard drive with checkpoints.")
    parser.add_argument("--device", type=str, default="cuda", required=False,
                        help="Use this to set the device the training should run with. "
                             "Available options are 'cuda', 'mps' and 'cpu'")
    parser.add_argument("--spark_master", type=str, default="local[*]", required=False,
                        help="[OPTIONAL] Specify a Spark master node")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def setup_training_arguments(args: Namespace):
    assert args.device in ['cpu', 'cuda', 'mps'], \
        f"--device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    if args.plans_identifier == "nnUNetPlans":
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(args.fold, str):
        if args.fold != "all":
            try:
                _ = int(args.fold)
            except ValueError as e:
                print(f"Unable to convert given value for fold to int: {args.fold}. "
                      "fold must be either 'all' or an integer!")
                raise e

    if args.val_best:
        assert not args.disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"


def build_trainer(dataset_name_or_id: Union[int, str],
                  configuration: str,
                  fold: int,
                  plans_identifier: str = 'nnUNetPlans',
                  use_compressed: bool = False,
                  device: torch.device = torch.device('cuda')):
    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    deepspeed_trainer = DeepSpeedTrainer(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json,
                                         unpack_dataset=not use_compressed, device=device)
    return deepspeed_trainer


def train(args):
    deepspeed_trainer = build_trainer()
    assert not (args.c and args.val), f'Cannot set --c and --val flag at the same time.'

    deepspeed_trainer.maybe_load_checkpoint(args.c, args.val, args.pretrained_weights)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if not args.val:
        deepspeed_trainer.run_training()
    if args.val_with_best:
        deepspeed_trainer.load_checkpoint(join(deepspeed_trainer.output_folder, 'checkpoint_best'))
    deepspeed_trainer.perform_actual_validation(args.npz)


if __name__ == "__main__":
    cmd_args = get_command_line_arguments()
    spark = SparkSession.builder \
        .appName("nnUNet DeepSpeed Spark Trainer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .master(cmd_args.spark_master) \
        .getOrCreate()

    deepspeed_config = deepspeed_base_config
    if hasattr(cmd_args, "deepspeed_config") and cmd_args.deepspeed_config is not None:
        deepspeed_config = cmd_args.deepspeed_config

    deepspeed_stage = 0
    if hasattr("zero_optimization", deepspeed_config) and deepspeed_config.zero_optimization is not None \
            and hasattr("stage",
                        deepspeed_config.zero_optimization) and deepspeed_config.zero_optimization.stage is not None:
        deepspeed_stage = int(deepspeed_config.zero_optimization.stage)

    if deepspeed_stage == 1:
        deepspeed_config = deepspeed_zero_1_config
    elif deepspeed_stage == 2:
        deepspeed_config = deepspeed_zero_2_config
    elif deepspeed_stage == 3:
        if hasattr("zero_optimizations", deepspeed_config) and deepspeed_config.zero_optimizations is not None \
                and hasattr("offload_param", deepspeed_config.zero_optimizations) \
                and deepspeed_config.zero_optimizations.offload_param is not None:
            deepspeed_config = deepspeed_zero_3_offload_config
        else:
            deepspeed_config = deepspeed_zero_3_config

    local_mode = cmd_args.spark_master.startswith("local")

    distributor = DeepspeedTorchDistributor(cmd_args.num_gpus, cmd_args.num_nodes, local_mode,
                                            deepspeedConfig=deepspeed_config)
    distributor.run(train, args=cmd_args)
    spark.stop()
