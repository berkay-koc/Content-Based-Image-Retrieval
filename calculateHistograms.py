import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

imageList = ['octopus', 'elephant', 'flamingo', 'kangaroo', 'leopards', 'sea_horse']

def saveHistograms(r_histogram, g_histogram, b_histogram, h_histogram):
    for i in range(0, 6):
        for j in range(1, 21):
            if (j < 10):
                path = 'train/{}/image_000{}'.format(imageList[i], j)
            else:
                path = 'train/{}/image_00{}'.format(imageList[i], j)
            if not os.path.exists(path):
                os.makedirs(path)
            if (j < 10):
                path = 'train/{}/image_000{}.jpg'.format(imageList[i], j)
            else:
                path = 'train/{}/image_00{}.jpg'.format(imageList[i], j)

            if(j < 10):
                np.savetxt('train/{}/image_000{}/r_histogram.csv'.format(imageList[i], j), r_histogram)
                np.savetxt('train/{}/image_000{}/g_histogram.csv'.format(imageList[i], j), g_histogram)
                np.savetxt('train/{}/image_000{}/b_histogram.csv'.format(imageList[i], j), b_histogram)
                np.savetxt('train/{}/image_000{}/h_histogram.csv'.format(imageList[i], j), h_histogram)
                print('appended', path)
            else:
                np.savetxt('train/{}/image_00{}/r_histogram.csv'.format(imageList[i], j), r_histogram)
                np.savetxt('train/{}/image_00{}/g_histogram.csv'.format(imageList[i], j), g_histogram)
                np.savetxt('train/{}/image_00{}/b_histogram.csv'.format(imageList[i], j), b_histogram)
                np.savetxt('train/{}/image_00{}/h_histogram.csv'.format(imageList[i], j), h_histogram)
                print('appended', path)

def calculateHistogram(im):
    im_rgb = im.convert("RGB")
    r_histogram = np.zeros(256, dtype=float)
    g_histogram = np.zeros(256, dtype=float)
    b_histogram = np.zeros(256, dtype=float)
    h_histogram = np.zeros(360, dtype=float)
    for k in range(0, im.width):
        for l in range(0, im.height):  # get histograms
            rgb_pixel_value = im_rgb.getpixel((k, l))
            r = rgb_pixel_value[0]
            g = rgb_pixel_value[1]
            b = rgb_pixel_value[2]

            cmax = max(r, g, b)
            cmin = min(r, g, b)
            diff = cmax - cmin

            x = im.height * im.width
            if cmax == cmin:
                h = 0
            elif cmax == r:
                h = (60 * ((g - b) / diff) + 360) % 360
            elif cmax == g:
                h = (60 * ((b - r) / diff) + 120) % 360
            elif cmax == b:
                h = (60 * ((r - g) / diff) + 240) % 360

            r_histogram[int(r)] += 1
            g_histogram[int(g)] += 1
            b_histogram[int(b)] += 1
            h_histogram[int(h)] += 1

    for m in range(0, 256):  # normalize histogram
        r_histogram[m] = float(r_histogram[m] / x)
        g_histogram[m] = float(g_histogram[m] / x)
        b_histogram[m] = float(b_histogram[m] / x)
    for n in range(0, 360):
        h_histogram[n] = float(h_histogram[n] / x)

    return r_histogram, g_histogram, b_histogram, h_histogram, x # returns the normalized histograms r g b h
