import logging
import shutil
from datetime import timedelta, datetime
from os.path import join, isdir
from time import time
from typing import List, Union

import deepspeed
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import empty_cache

log = logging.getLogger(__name__)


class DeepSpeedTrainer(nnUNetTrainer):
    model_engine = None
    checkpoint_path = None

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 1000, deepspeed_stage: str = 'stage_1',
                 args: dict = None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        log.info("Initializing the trainer")
        self.num_epochs = num_epochs
        self.args = args


    def initialize(self):
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            self.loss = self._build_loss()
            if not torch.__version__.startswith("2.2") and self._do_i_compile():
                self.loss = torch.compile(self.loss)
            self.was_initialized = True
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def run_training(self):
        # Model construction, and loading dataset has to be done before deepspeed init call
        self.get_dataloaders()
        self.initialize()

        deepspeed.init_distributed(timeout=timedelta(minutes=5))
        if self.local_rank == -1:
            self.local_rank = deepspeed.dist.get_rank()

        self.model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.network,
            model_parameters=self.network.parameters(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )

        start = datetime.now()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()

            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                next_batch = next(self.dataloader_train)
                loss = self.model_engine(next_batch)
                self.model_engine.backward(loss)
                self.model_engine.step()
                train_outputs.append(loss)
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    next_input = next(self.dataloader_val)
                    validation_step_output = self.validation_step(next_input)
                    val_outputs.append(validation_step_output)
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

        if self.local_rank == 0:
            print("Training completed in: " + str(datetime.now() - start))

    def on_train_end(self) -> None:
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
        else:
            self.current_epoch -= 1
            self.model_engine.save_checkpoint(self.output_folder, "checkpoint_final")
            self.current_epoch += 1

            # now we can delete latest
            final_checkpoint_path = join(self.output_folder, "checkpoint_latest")
            if self.local_rank == 0 and isdir(final_checkpoint_path):
                shutil.rmtree(final_checkpoint_path)

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.model_engine.save_checkpoint(self.output_folder, "checkpoint_latest")

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.model_engine.save_checkpoint(self.output_folder, "checkpoint_best")

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    # TODO ensure the checks for checkpoint validations are consistent with Deepspeed engine's conventions
    def maybe_load_checkpoint(self, continue_training: bool, validation_only: bool, pretrained_weights_tag: str = None):
        if continue_training and pretrained_weights_tag is not None:
            raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                               'be used at the beginning of the training.')
        if continue_training:
            expected_checkpoint_tag = "checkpoint_final"
            if not isdir(expected_checkpoint_tag):
                expected_checkpoint_tag = "checkpoint_latest"
            # special case where --c is used to run a previously aborted validation
            if not isdir(expected_checkpoint_tag):
                expected_checkpoint_tag = "checkpoint_best"
            if not isdir(expected_checkpoint_tag):
                print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                      f"continue from. Starting a new training...")
                expected_checkpoint_tag = None
        elif validation_only:
            expected_checkpoint_tag = "checkpoint_final"
            if not isdir(expected_checkpoint_tag):
                raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
        else:
            if pretrained_weights_tag is not None:
                if not self.was_initialized:
                    self.initialize()
                self.model_engine.load_checkpoint(
                    self.output_folder,
                    tag=pretrained_weights_tag,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                    load_module_only=True
                )
            expected_checkpoint_tag = None

        if expected_checkpoint_tag is not None:
            self.model_engine.load_checkpoint(self.output_folder, expected_checkpoint_tag)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        deepspeed.comm.get_world_size()
        world_size = deepspeed.dist.get_world_size()

        tps = [None for _ in range(world_size)]
        deepspeed.dist.all_gather_object(tps, tp)
        tp = np.vstack([i[None] for i in tps]).sum(0)

        fps = [None for _ in range(world_size)]
        deepspeed.dist.all_gather_object(fps, fp)
        fp = np.vstack([i[None] for i in fps]).sum(0)

        fns = [None for _ in range(world_size)]
        deepspeed.dist.all_gather_object(fns, fn)
        fn = np.vstack([i[None] for i in fns]).sum(0)

        losses_val = [None for _ in range(world_size)]
        deepspeed.dist.all_gather_object(losses_val, outputs_collated['loss'])
        loss_here = np.vstack(losses_val).mean()

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if isinstance(filename_or_checkpoint, str):
            self.model_engine.load_checkpoint(filename_or_checkpoint)
