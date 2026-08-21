# lightning_focus.py
# Modeling specified for pytorch-lightning

import gc
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor, Tensor

from ....utils import get_logger
from ....utils.abstracts import LightningModelInputs, LightningTrainingArguments
from .collation_focus import FocusCollator
from .configuration_focus import FocusConfig
from .modeling_focus import FocusForClassification

logger = get_logger(__name__)


@dataclass
class FocusLightningTrainingArguments(LightningTrainingArguments):
  """
  FocusLightningTrainingArguments is a subclass of the LightningTrainingArguments,
   works to store specified parameters or arguments for the Training of FocusModel.
  """

  def collate(self):
    # TODO: this method will run automatically, clarify its logic, currently we account it to something like `__post_init__`, awaiting check
    pass


@dataclass
class FocusLightningModelInputs(LightningModelInputs):
  """
  FocusLightningModelInputs is a subclass of the LightningModelInputs,
   works to store specified parameters or arguments for the Training of FocusModel.
  """

  cls_labels: Optional[LongTensor] = field(
    default=None,
    metadata={"note": "Labels for judge classification & confidence."},
  )


class FocusLightningForClassification(pl.LightningModule):
  module_name = "FocusLightningForClassification"

  def __init__(
    self,
    config: FocusConfig,
    collator: Optional[FocusCollator] = None,
    example_input_array_shape: Optional[Union[torch.Size, Tuple, List]] = None,
  ):
    super().__init__()
    self.config = config
    self.collator = collator
    # Given self.model is an nn.Module, self.parameters() will find its parameters.
    self.model = FocusForClassification(config, collator)

    # --- Hyperparameters to modify directly --- #
    # Optimizer settings
    self.target_lr = 1e-3
    self.min_lr = 8e-5
    self.optimizer_weight_decay = 5e-4

    # Main scheduler settings (ReduceLROnPlateau)
    self.scheduler_patience = 16
    self.scheduler_factor = 0.8

    # Warmup settings
    self.warmup_enabled = True
    self.warmup_epochs = 10
    self.warmup_initial_lr_factor = 0.1  # Start at 10% of the target LR
    # -------------------------------------------

    # Save these hyperparameters to the log
    self.save_hyperparameters(ignore=["collator"])
    # self.save_hyperparameters("target_lr", "warmup_epochs", "scheduler_patience") <- this leads to key error for that the keys are not provided as arguments

    # FIXME: our model has problems for torch.jit to trace
    # so here we will temporarily ignore the assignment for `example_input_array` here with an early return
    # return

    # The TensorBoard Logger, tried to create a visual diagram of the model's architecture (the 'computational graph')
    # to display in the 'Graphs' tab in TensorBoard. To do this, TensorBoard Logger needs a sample input tensor
    # to trace its path through your model.
    if example_input_array_shape is not None:
      logger.info(
        f"FocusLightningForClassification.__init__() - creating example_input_array of shape: {example_input_array_shape}"
      )
      self.example_input_array = torch.zeros(
        example_input_array_shape, requires_grad=False, dtype=torch.float32
      )
      logger.info(
        f"FocusLightningForClassification.__init__() - creating example_input_array of shape: {self.example_input_array.size()}"
      )
    else:
      try:
        self.example_input_array = torch.zeros(
          [1, config.seq_len, config.hidden_size],
          requires_grad=False,
          dtype=torch.float32,
          # correct shape is [1, num_channels, height, width]
        )
      except Exception as e:
        warnings.warn(
          f"FocusLightningForClassification.__init__() - failed to create `example_input_array` with provided arguments. - Message: {e}\n"
          "If you are not using `TensorBoard Logger`, please ignore this warning."
        )

    # TODO: init weights & final processing

  def set_example_input_array(
    self, example_input_array_shape: Optional[Union[torch.Size, Tuple, List]]
  ):
    # this is not correct, the `example_input_array` related logic is launched within the __init__()
    self.example_input_array = torch.zeros(
      example_input_array_shape, requires_grad=False, dtype=torch.float32
    )

  def forward(self, *args, **kwargs):
    """
    Inherits from the parent core FocusModel for the use of TensorBoard 'computational graph'

    Defines the forward pass for the LightningModule.
    It simply delegates the call to the actual model.
    """
    return self.model(*args, **kwargs)

  def training_step(
    self, batch_inputs: Union[FocusLightningModelInputs, Tuple, List], batch_idx
  ) -> Union[FloatTensor, None]:
    # FIXME: `debug_once` with variables has bugs, fallback to use `batch_idx`
    if batch_idx == 0:
      logger.debug_once(
        f"FocusLightningForClassification.training_step() - see what you get from `batch_inputs`:\n{batch_inputs}"
      )

    # declare `pixel_values` & `cls_labels` preliminarily
    cls_labels = None
    pixel_values = None

    if isinstance(batch_inputs, FocusLightningModelInputs):
      # for normal cases
      pixel_values = batch_inputs.pixel_values
      cls_labels = batch_inputs.cls_labels
    elif isinstance(batch_inputs, (Tuple, List)):
      # TODO: temporal solution, migrate this to dataloader collate function
      pixel_values = batch_inputs[0]
      targets = batch_inputs[1]
      cls_labels = torch.stack([_["labels"][0] for _ in targets], dim=0)

      if batch_idx == 0:
        logger.debug_once(
          f"FocusLightningForClassification.training_step() - images type: {pixel_values.dtype} - shape: {pixel_values.size()}"
        )
        logger.debug_once(
          f"FocusLightningForClassification.training_step() - targets length: {len(targets)}"
        )
        logger.debug_once(
          f"FocusLightningForClassification.training_step() - cls_labels type: {cls_labels.dtype} - size: {cls_labels.size()}"
        )

    if pixel_values is None or cls_labels is None:
      warnings.warn(
        "FocusLightningForClassification.training_step() gets NoneType `pixel_values` or `cls_labels`."
      )
      return None

    loss, (cls_loss, con_loss), (cls_logits, con_logits) = self(
      pixel_values, cls_labels
    )

    batch_size = pixel_values.shape[0]
    self.log(
      "train_loss",
      loss,
      batch_size=batch_size,
      on_step=True,
      on_epoch=True,
      prog_bar=True,
    )
    return loss

  def on_train_epoch_start(self):
    """
    This hook is called at the beginning of each training epoch.
    We use it to manually implement the linear learning rate warmup.
    """
    if not self.warmup_enabled or self.current_epoch > self.warmup_epochs:
      # Skip if warmup is disabled or we are past the warmup phase

      # Log the current learning rate to see it in TensorBoard/logs
      self.log(
        "learning_rate",
        self.optimizers().param_groups[0]["lr"],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
      )
      return

    # Get the optimizer
    optimizer = self.optimizers()

    # Calculate the initial and target learning rates
    initial_lr = self.target_lr * self.warmup_initial_lr_factor

    # Calculate the current learning rate using a linear ramp-up
    # (self.current_epoch is 0-indexed)
    progress = self.current_epoch / self.warmup_epochs
    current_lr = initial_lr + progress * (self.target_lr - initial_lr)

    # Manually set the learning rate for all parameter groups
    for param_group in optimizer.param_groups:
      param_group["lr"] = current_lr

    # Log the current learning rate to see it in TensorBoard/logs
    self.log("learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=True)

  def on_train_epoch_end(self):
    torch.cuda.empty_cache()
    gc.collect()

  def configure_optimizers(self):
    """
    Set up the optimizer and the main scheduler.
    The warmup logic is handled separately in `on_train_epoch_start`.

    ## What is ReduceLROnPlateau?
      Think of it as the "Patient Parent" Scheduler.

      How it works: It watches a metric you choose (almost always val_loss). It does nothing as long as the metric is improving.

      Patience: If the val_loss stops improving for a certain number of epochs (the patience value), the "parent" gets impatient.

      Action: After its patience runs out, it reduces the learning rate by a factor (e.g., multiplies it by 0.1, making it 10x smaller).

      Goal: The idea is that if you're stuck in a "plateau" (not making progress), a smaller learning rate might help you find a better path downhill.

      Why it can be tricky: It's reactive, not proactive. It completely depends on your validation data being stable. Sometimes it can reduce the LR too early or too late.
    """
    # Create the optimizer
    optimizer = optim.Adam(
      self.parameters(),
      lr=self.target_lr,
      weight_decay=self.optimizer_weight_decay,
    )

    # Create the main scheduler
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      "min",
      patience=self.scheduler_patience,
      factor=self.scheduler_factor,
      min_lr=self.min_lr,
    )

    # Return them in the format Lightning expects for metric-based schedulers
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": main_scheduler,
        "monitor": "val_loss",  # The key value to monitor
        "interval": "epoch",
        "frequency": 1,
      },
    }

  def set_optimizers(self, configure_optimizers: Callable):
    """An External API for the Configuration of Optimizers"""
    if isinstance(configure_optimizers, Callable):
      self.configure_optimizers = configure_optimizers
    else:
      raise TypeError(
        "FocusLightningForClassification.set_optimizers() - expected type of provided argument `configure_optimizers` is Callable, "
        f"but got {type(configure_optimizers)}."
      )

  def validation_step(
    self, batch_inputs: Union[FocusLightningModelInputs, Tuple], batch_idx
  ):
    # FIXME: `debug_once` with variables has bugs, fallback to use `batch_idx`
    if batch_idx == 0:
      logger.debug_once(
        f"FocusLightningForClassification.train_step() - see what you get from `batch_inputs`:\n{batch_inputs}"
      )

    # declare `pixel_values` & `cls_labels` preliminarily
    cls_labels = None
    pixel_values = None

    if isinstance(batch_inputs, FocusLightningModelInputs):
      # for normal cases
      pixel_values = batch_inputs.pixel_values
      cls_labels = batch_inputs.cls_labels
    elif isinstance(batch_inputs, (Tuple, List)):
      # TODO: temporal solution, migrate this to dataloader collate function
      pixel_values = batch_inputs[0]
      targets = batch_inputs[1]
      cls_labels = torch.stack([_["labels"][0] for _ in targets], dim=0)

      if batch_idx == 0:
        logger.debug_once(
          f"FocusLightningForClassification.validation_step() - images type: {pixel_values.type} - shape: {pixel_values.shape}"
        )
        logger.debug_once(
          f"FocusLightningClassification.validation_step() - targets length: {len(targets)}"
        )
        logger.debug_once(
          f"FocusLightningForClassification.validation_step() - cls_labels type: {cls_labels.type} - size: {cls_labels.size()}"
        )

    if pixel_values is None or cls_labels is None:
      warnings.warn(
        "FocusLightningForClassification.validation_step() gets NoneType `pixel_values` or `cls_labels`."
      )
      return None

    loss, (cls_loss, con_loss), (cls_logits, con_logits) = self.model(
      pixel_values, cls_labels
    )

    batch_size = pixel_values.shape[0]
    self.log(
      "val_loss",
      loss,
      batch_size=batch_size,
      on_step=False,
      on_epoch=True,
      prog_bar=True,
    )

  def on_validation_epoch_end(self):
    torch.cuda.empty_cache()
    gc.collect()

  @torch.no_grad()
  def test_step(
    self, batch_inputs: Union[FocusLightningModelInputs, Tuple], batch_idx
  ) -> Union[FloatTensor, None]:
    # FIXME: `debug_once` with variables has bugs, fallback to use `batch_idx`
    if batch_idx == 0:
      logger.debug_once(
        f"FocusLightningForClassification - test_step() - see what you get from `batch_inputs`:\n{batch_inputs}"
      )

    # declare `pixel_values` & `cls_labels` preliminarily
    cls_labels = None
    pixel_values = None

    if isinstance(batch_inputs, FocusLightningModelInputs):
      # for normal cases
      pixel_values = batch_inputs.pixel_values
      cls_labels = batch_inputs.cls_labels
    elif isinstance(batch_inputs, (Tuple, List)):
      # TODO: temporal solution, migrate this to dataloader collate function
      pixel_values = batch_inputs[0]
      targets = batch_inputs[1]
      cls_labels = torch.stack([_["labels"][0] for _ in targets], dim=0)

      if batch_idx == 0:
        logger.debug_once(
          f"FocusLightningForClassification - test_step() - images type: {pixel_values.dtype} - shape: {pixel_values.size()}"
        )
        logger.debug_once(
          f"FocusLightningForClassification - test_step() - targets length: {len(targets)}"
        )
        logger.debug_once(
          f"FocusLightningForClassification - test_step() - cls_labels - type: {cls_labels.dtype} - size: {cls_labels.size()}"
        )

    if pixel_values is None or cls_labels is None:
      warnings.warn(
        "FocusLightningForClassification.test-step() gets NoneType `pixel_values` or `cls_labels`."
      )
      return None

    loss, (cls_loss, con_loss), (cls_logits, con_logits) = self.model(
      pixel_values, cls_labels
    )

    batch_size = pixel_values.shape[0]
    self.log(
      "train_loss",
      loss,
      batch_size=batch_size,
      on_step=True,
      on_epoch=True,
      prog_bar=True,
    )
    return loss

  def on_test_epoch_end(self):
    torch.cuda.empty_cache()
    gc.collect()

  @torch.no_grad()
  def classify(self, pixel_values: FloatTensor) -> Tuple[Tensor, Tensor]:
    # cls_results: [batch_size,] <- classification results (range[num_classes])
    # focus_window_idx: [batch_size,] <- row-majored window index (range[num_windows])
    cls_results, focus_window_idx_lst = self.model.classify(pixel_values)
    logger.debug(
      f"FocusLightningForClassification.classify()\n - cls_results:\n{cls_results}\n - focus_window_idx:\n{focus_window_idx_lst[:8]}"
    )

    return cls_results, focus_window_idx_lst


# lightning_focus.py ends here
