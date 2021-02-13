import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image
import re
import sys
from deephistopath.wsi import util
from deephistopath.wsi.util import Time
#from tempfile import TemporaryFile

the_name = sys.argv[1]

image_svs = './sys_' + the_name + '/'
image_png = './png_' + the_name + '/'
image_npy = './npy_' + the_name + '/'

os.system('mkdir -p ' + image_png)
os.system('mkdir -p ' + image_npy)

remain = [image_svs + s for s in os.listdir(image_svs)]

SCALE_FACTOR = 32

output = open('dim_' + the_name + '.txt', 'w')

#image_jpg = '/local/disk4/likeyu/data/training_slides_png_test_2'
#image_npy = '/local/disk4/likeyu/data/training_slides_npy_test_2'
#output = '/local/disk4/likeyu/data/output.csv'
#
#SCALE_FACTOR = 32
#F = '/local/disk4/likeyu/data/missingfile.lst'
#remain = []
#f = open(F, 'r')
#for i in f:
#    link = i.strip().split('\t')
#    remain.append(link[0])
#o = open(output, 'w')

def open_slide(filename):
    """
    Open a whole-slide image (*.svs, etc).

    Args:
        filename: Name of the slide file.

    Returns:
        An OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


def slide_to_scaled_pil_image(slide_filepath):
    
    """
    Convert a WSI training slide to a scaled-down PIL image.

    Args:
        slide_filepath.

    Returns:
        Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    print("Opening Slide %s" % (slide_filepath))
    slide = open_slide(slide_filepath)

    large_w, large_h = slide.dimensions
   # print('large_w, large_h ', large_w, large_h)
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
   # print('new_w, new_h ', new_w, new_h)
    level = 0
    #print('level ', level)

    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)

    slide_name = os.path.basename(slide_filepath).split('.')[0]
    img_path = os.path.join(image_png, slide_name + '.png')
    print("Saving image to: " + img_path)
    img.save(img_path)
    output.write(slide_filepath + '\t'+ str(new_w)+ '\t'+ str(new_h)+ '\t'+ str(new_w*new_h)+ '\n')
    return img, large_w, large_h, new_w, new_h

def slide_to_scaled_np_image(slide_filepath):
    """
    Convert a WSI training slide to a scaled-down NumPy image.

    Args:
        slide_filepath

    Returns:
        Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
    """
    pil_img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_filepath)
    slide_name = os.path.basename(slide_filepath).split('.')[0]
    npy_path = os.path.join(image_npy, slide_name + '.npy')

    np_img = util.pil_to_np_rgb(pil_img)
    np.save(npy_path, np_img)
    print("Saving npy to: " + npy_path)


def training_slide_range_to_images(start_ind, end_ind):
    """
    Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

    Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).

    Returns:
    The starting index and the ending index of the slides that were converted.
    """
    imagefile_path = []
    allfiles = os.listdir(image_svs)
    for images in allfiles:
        image_path = os.path.join(image_svs, images)
        imagefile_path.append(image_path)

    for slide_num in range(start_ind, end_ind + 1):
        slide_to_scaled_np_image(remain[slide_num-1])
    return (start_ind, end_ind)

def multiprocess_training_slides_to_images():
    
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = Time()

    # how many processes to use
    num_processes = 5
    pool = multiprocessing.Pool(num_processes)

    #num_train_images = 24
    num_train_images = len(remain)
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    # each task specifies a range of slides
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((start_index, end_index))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(training_slide_range_to_images, t))

    for result in results:
        (start_ind, end_ind) = result.get()
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    timer.elapsed_display()


def main():
    multiprocess_training_slides_to_images()
    output.close()
    return


main()
