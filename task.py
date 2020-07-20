# -*- coding: utf-8 -*-

import logging
from nptyping import NDArray
from typing import List, Any
import os
import sys
from skimage.io import imsave
from skimage.io import imread
import numpy as np
from dataclasses import dataclass
import getopt

ListStr = List[str]
Image = NDArray[(Any, Any), int]
ImageStack = NDArray[(Any, Any, Any), int]


@dataclass
class Point3D:
    """Form a 3D point"""
    x: float
    y: float
    z: float


# Hard coded values:
spacing = Point3D(0.97, 0.97, 1.6)  # spacing between pixels
log_file_name = "log.txt"  # name of file for the log output
result_file_name = "result.txt"  # name of file for the result
mask_threshold = 0.7  # threshold for masking


def uploadImages(input_dir: str, ext: str = ".tif") -> ImageStack:
    """Collect file paths from the directory and upload images"""
    if not input_dir:
        raise Exception("uploadImages: input directory is not provided")
    file_paths: ListStr = []
    for (dir_path, dir_names, file_names) in os.walk(input_dir):
        for i in range(0, len(file_names)):
            if file_names[i].endswith(ext):
                file_paths.append(input_dir+os.sep+file_names[i])
                logger.info("Found file: %s", file_paths[-1])
    if not file_paths:
        raise Exception("No files with extension " + ext + " found")
    stack: ImageStack = np.stack([imread(p, as_gray=True).astype(int)
                                  for p in file_paths], axis=-1)
    return stack


def saveMaskImages(output_dir: str, mask: ImageStack):
    """Save image wise stack to the diractory"""
    if not output_dir:
        raise Exception("saveMaskImages: output directory is not provided")
    if mask.size == 0:
        raise Exception("saveMaskImages: mask is empty")
    for i in range(mask.shape[2]):
        path = output_dir+os.sep+"mask_"+str(i)+".png"
        logger.info("Saving mask to file " + path)
        imsave(path, mask[:, :, i])


class Scan3D():

    def __init__(self, image_stack: ImageStack, spacing: Point3D):
        self._image_stack = image_stack
        self._spacing = spacing

    def getMaxIntensity(self) -> int:
        return np.max(self._image_stack)

    def getMeanIntensity(self) -> int:
        return np.mean(self._image_stack)

    def getCenterImageVolume(self) -> Point3D:
        """Return the center of the image volume"""
        shape = self._image_stack.shape
        center = Point3D(0, 0, 0)
        center.x = (shape[1]-1)/2*self._spacing.x
        center.y = (shape[0]-1)/2*self._spacing.y
        center.z = (shape[2]-1)/2*self._spacing.z
        return center

    def getMetalObjectMask(self,
                           threshold: float = mask_threshold) -> ImageStack:
        """Return mask of the images based on the threshold
        Here, image intensity get normolized to be <=1
        """
        image_stack_normolized = self._image_stack / self.getMaxIntensity()
        mask = np.where(image_stack_normolized < threshold, 0, 1).astype(int)
        return mask


if __name__ == '__main__':
    try:
        logger = logging.getLogger(__name__)

        options, remainder = getopt.getopt(
            sys.argv[1:],
            'i:o',
            ['input=', 'output=']
            )

        for opt, arg in options:
            if opt in ('-i', '--input'):
                input_dir = arg
            if opt in ('-o', '--output'):
                output_dir = arg
        if not os.path.exists(input_dir):
            raise Exception("nput directory does not exist")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging_file_path = output_dir+os.sep+log_file_name
        result_file_path = output_dir+os.sep+result_file_name
        logging.basicConfig(filename=logging_file_path, level=logging.INFO)

        image_stack = uploadImages(input_dir)
        scan3D = Scan3D(image_stack, spacing)
        f = open(result_file_path, "w")
        f.write("max intensity="
                + str(scan3D.getMaxIntensity())
                + "; mean intensity="
                + str(scan3D.getMeanIntensity())+"\n")
        center = scan3D.getCenterImageVolume()
        f.write("volume center: x=" + str(center.x)
                + "; y=" + str(center.y)
                + "; z=" + str(center.z))
        f.close()
        masks = scan3D.getMetalObjectMask()
        saveMaskImages(output_dir, masks)
    except Exception as e:
        logger.error(str(e))

 
