# tools.py

import os
from typing import Optional


def find_latest_checkpoint_for_model(
  checkpoint_dir: str, model_module_name: str
) -> Optional[str]:
  """
  Finds the latest checkpoint file for a given model in a directory.

  The function looks for .ckpt files containing the `model_module_name` in their filename,
  and returns the path to the most recently modified one.
  """
  if not os.path.isdir(checkpoint_dir):
    return None

  all_ckpts_for_model = [
    os.path.join(checkpoint_dir, f)
    for f in os.listdir(checkpoint_dir)
    if f.endswith(".ckpt") and model_module_name in f
  ]

  if not all_ckpts_for_model:
    return None

  all_ckpts_for_model.sort(key=os.path.getmtime, reverse=True)
  return all_ckpts_for_model[0]


# tools.py
