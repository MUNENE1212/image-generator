"""Binary storage for selfies, generated images, and trained LoRAs."""

from image_generator.storage.base import Storage
from image_generator.storage.local import LocalStorage

__all__ = ["LocalStorage", "Storage"]
