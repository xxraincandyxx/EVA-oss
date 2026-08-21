# structs.py

from dataclasses import dataclass, field
from typing import List, Optional

# ------------------- #
# --- Dataclasses --- #
# ------------------- #


@dataclass
class FrameInfo:
  desc: Optional[str] = field(default=None)
  direct_vec: Optional[List[float]] = field(default=None)
  positi_vec: Optional[List[float]] = field(default=None)


# structs.py ends here
