from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Krit Patyarath"

from typing import TYPE_CHECKING

import imagehash

if TYPE_CHECKING:
    from typing import Any

    from imagehash import ImageHash
    from PIL.Image import Image

    import miavisc as mv


class FrameHashPool:
    def __init__(
        self, setting: mv.settings.HashSetting, *, keep_unique_data: bool = False
    ) -> None:
        self._setting = setting
        self._hash_pool: list[ImageHash] = []

        self._keep_unique_data: bool = keep_unique_data

        if self._keep_unique_data:
            self._data_pool: list[Any] = []

    @property
    def hash_pool(self) -> list[ImageHash]:
        return self._hash_pool

    @property
    def data_pool(self) -> list[Any]:
        return self._data_pool

    def _similar_prev_hashes(self, image_hash: ImageHash) -> bool:
        # similar hashes should be in the back, so search in reverse.
        if self._setting.history_size == 0:
            return False
        for i, prev_hash in enumerate(reversed(self._hash_pool)):
            if self._setting.history_size > 0 and i >= self._setting.history_size:
                return False
            if prev_hash - image_hash <= self._setting.threshold:
                return True
        return False

    def add_image_if_unique(self, image: Image, data: Any = None) -> None:
        image_hash = imagehash.dhash(image, hash_size=self._setting.size)
        if self._similar_prev_hashes(image_hash):
            return

        self._hash_pool.append(image_hash)
        if self._keep_unique_data:
            self._data_pool.append(data)
