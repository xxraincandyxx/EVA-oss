# flasher.py

import os

from ..utils import get_logger

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #


logger = get_logger()


# -------------------- #
# --- Main Classes --- #
# -------------------- #


class EvaFlasher:
  """Control for the Flasher"""

  def __init__(self, gpio_addr: str = "492"):
    self.is_flasher_available = False
    self.gpio_addr = gpio_addr
    self.release()
    self.export(gpio_addr=gpio_addr)

  def export(self, gpio_addr: str):
    try:
      os.system(f"echo {gpio_addr} > /sys/class/gpio/export")
      os.system(f"echo out > /sys/class/gpio/gpio{gpio_addr}/direction")
      self.is_flasher_available = True
      logger.info(f"Flasher.export() - export {gpio_addr} successfully.")

    except Exception as e:
      logger.info(f"Flasher.export() - export {gpio_addr} failed. Message: {e}")

  def unexport(self, gpio_addr: str):
    os.system(f"echo {gpio_addr} > /sys/class/gpio/unexport")
    self.is_flasher_available = False
    logger.info(f"Flasher.unexport() - unexport {gpio_addr} successfully.")

  def flash_on(self):
    if not self.is_flasher_available:
      logger.warning("Flasher.flash_on() - Fatal: flasher not available.")
      return

    os.system(f"echo 1 > /sys/class/gpio/gpio{self.gpio_addr}/value")
    self.cli_log()

  def flash_off(self):
    if not self.is_flasher_available:
      logger.warning("Flasher.flash_off() - Fatal: flasher not available.")
      return

    os.system(f"echo 0 > /sys/class/gpio/gpio{self.gpio_addr}/value")
    self.cli_log()

  def cli_log(self):
    if not self.is_flasher_available:
      logger.warning("Flasher.cli_log() - Fatal: flasher not available.")

    print("--- GPIO Value ---")
    os.system(f"cat /sys/class/gpio/gpio{self.gpio_addr}/value")
    print("------------------")

  def release(self):
    self.flash_off()
    self.unexport(gpio_addr=self.gpio_addr)


# flasher.py ends here
