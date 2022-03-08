import math
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

from dt_computer_vision.camera import BGRImage

Color = Union[Tuple[float], List[float]]


class AntiInstagram:

    def __init__(self,
                 image_scale: float = 0.2,
                 color_balance_scale: float = 0.8,
                 lower_threshold: Optional[Color] = None,
                 higher_threshold: Optional[Color] = None):
        # store parameters
        self.color_balance_scale = color_balance_scale
        self.image_scale = image_scale
        self.lower_threshold = lower_threshold or [0, 0, 0]
        self.higher_threshold = higher_threshold or [255, 255, 255]

    def update(self, image: BGRImage):
        # scale image (if necessary)
        if self.image_scale != 1.0:
            image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)

        # TODO: this cropping seems arbitrary and Duckiebot-specific, if needed, pass it as a `roi` argument
        # H = image.shape[0]
        # image = image[int(H * 0.3): (H - 1), :, :]

        half_percent = self.color_balance_scale / 2
        channels = cv2.split(image)

        lower_threshold = []
        for idx, channel in enumerate(channels):
            # find the low and high percentile values (based on the input percentile)
            height, width = channel.shape
            num_pixels = width * height
            flattened = channel.reshape(num_pixels)
            # sort entries
            flattened = np.sort(flattened)
            # calculate thresholds
            lower_threshold.append(flattened[int(math.floor(num_pixels * half_percent))])
        # update lower threshold
        self.lower_threshold = lower_threshold

    def apply(self,
              image: BGRImage,
              image_scale: float = 1.0,
              lower_threshold: Optional[Color] = None,
              higher_threshold: Optional[Color] = None):
        lower_thr = lower_threshold or self.lower_threshold
        higher_thr = higher_threshold or self.higher_threshold
        # scale image (if necessary)
        if image_scale != 1.0:
            image = cv2.resize(image, (0, 0), fx=image_scale, fy=image_scale)
        # split BGR -> B, G, R
        channels = cv2.split(image)
        out_channels = []
        # saturate each channel
        for idx, channel in enumerate(channels):
            thresholded = self._apply_threshold(channel, lower_thr[idx], higher_thr[idx])
            # TODO: that `.copy()` might be expensive
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)
        # recombine B, G, R -> BGR
        return cv2.merge(out_channels)

    @classmethod
    def _apply_threshold(cls, channel, low_value, high_value):
        # create mask for pixels below threshold
        low_mask = channel < low_value
        # saturate highlighted pixels with low value
        channel = cls._apply_mask(channel, low_mask, low_value)
        # create mask for pixels above threshold
        high_mask = channel > high_value
        # saturate highlighted pixels with high value
        channel = cls._apply_mask(channel, high_mask, high_value)
        # return saturated channel
        return channel

    @classmethod
    def _apply_mask(cls, matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()
