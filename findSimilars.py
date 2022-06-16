import numpy as np
import os
from PIL import Image
from math import sqrt
from matplotlib import pyplot as plt
from calculateHistograms import calculateHistogram
from time import sleep

def concatImagesHorizontal(im1, im2, im3, im4, im5, im6, color=(255, 255, 255)):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width + im4.width + im5.width + im6.width, max(im1.height, im2.height, im3.height, im4.height, im5.height, im6.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    dst.paste(im4, (im1.width + im2.width + im3.width, 0))
    dst.paste(im5, (im1.width + im2.width + im3.width + im4.width, 0))
    dst.paste(im6, (im1.width + im2.width + im3.width + im4.width + im5.width, 0))
    return dst

def concatImagesVertical(im1, im2):
    if(im2.width > im1.width):
        dst = Image.new('RGB', (im2.width , im1.height + im2.height))
    else:
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def euclideanDistance(val1, val2):
    return (val1-val2)**2

def findSimilars():
    imageList = ['elephant', 'octopus', 'flamingo', 'kangaroo', 'leopards', 'sea_horse']
    ctr = 0
    totalRgbSuccess = 0
    totalHueSuccess = 0
    outputPath = 'outputs'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    for i in range(0, 6):
        successRateRgb = 0
        successRateHue = 0
        for j in range(21, 31):
            rgb_list = []
            hue_list = []
            testPath = 'test/{}/image_00{}.jpg'.format(imageList[i], j)
            im_test = Image.open(testPath)
            r_histogram_test, g_histogram_test, b_histogram_test, h_histogram_test, x = calculateHistogram(im_test)  # histograms of test img
            for k in range(0, 6):
                for l in range(1,21):
                    if (l < 10):
                        red_histogram_path = 'train/{}/image_000{}/r_histogram.csv'.format(imageList[k], l)
                        green_histogram_path = 'train/{}/image_000{}/g_histogram.csv'.format(imageList[k], l)
                        blue_histogram_path = 'train/{}/image_000{}/b_histogram.csv'.format(imageList[k], l)
                        hue_histogram_path = 'train/{}/image_000{}/h_histogram.csv'.format(imageList[k], l)
                    else:
                        red_histogram_path = 'train/{}/image_00{}/r_histogram.csv'.format(imageList[k], l)
                        green_histogram_path = 'train/{}/image_00{}/g_histogram.csv'.format(imageList[k], l)
                        blue_histogram_path = 'train/{}/image_00{}/b_histogram.csv'.format(imageList[k], l)
                        hue_histogram_path = 'train/{}/image_00{}/h_histogram.csv'.format(imageList[k], l)

                    r_histogram_train = np.genfromtxt(red_histogram_path, delimiter=',')
                    g_histogram_train = np.genfromtxt(green_histogram_path, delimiter=',')
                    b_histogram_train = np.genfromtxt(blue_histogram_path, delimiter=',')
                    h_histogram_train = np.genfromtxt(hue_histogram_path, delimiter=',')
                    rgb_sum = 0
                    h_sum = 0
                    for m in range(0, 256):
                        rgb_sum += euclideanDistance(r_histogram_train[m], r_histogram_test[m])
                        rgb_sum += euclideanDistance(g_histogram_train[m], g_histogram_test[m])
                        rgb_sum += euclideanDistance(b_histogram_train[m], b_histogram_test[m])

                    for n in range(0, 360):
                        h_sum += euclideanDistance(h_histogram_train[n], h_histogram_test[n])

                    rgb_list.append(sqrt(rgb_sum))
                    hue_list.append(sqrt(h_sum))
                    '''print(testPath)
                    print(red_histogram_path)
                    print(len(rgb_list))'''
            min1 = 0
            min2 = 1
            min3 = 2
            min4 = 3
            min5 = 4
            for a in range(5, 120):
                if rgb_list[a] < rgb_list[min1]:
                    min1 = a
                elif rgb_list[a] < rgb_list[min2]:
                    min2 = a
                elif rgb_list[a] < rgb_list[min3]:
                    min3 = a
                elif rgb_list[a] < rgb_list[min4]:
                    min4 = a
                elif rgb_list[a] < rgb_list[min5]:
                    min5 = a
            if imageList[i] == imageList[int(min1/20)] or imageList[i] == imageList[int(min2/20)] or imageList[i] == imageList[int(min3/20)] or imageList[i] == imageList[int(min4/20)] or imageList[i] == imageList[int(min5/20)]:
                successRateRgb += 1
            if(((min1+1) % 20) < 10):
                if(((min1+1) % 20) == 0):
                    path1 = 'train/{}/image_00{}.jpg'.format(imageList[int(min1/20)], ((min1+1) % 20) + 20)
                else:
                    path1 = 'train/{}/image_000{}.jpg'.format(imageList[int(min1 / 20)], ((min1 + 1) % 20))
            else:
                path1 = 'train/{}/image_00{}.jpg'.format(imageList[int(min1 / 20)], (min1 + 1) % 20)

            if (((min2 + 1) % 20) < 10):
                if (((min2 + 1) % 20) == 0):
                    path2 = 'train/{}/image_00{}.jpg'.format(imageList[int(min2 / 20)], ((min2 + 1) % 20) + 20)
                else:
                    path2 = 'train/{}/image_000{}.jpg'.format(imageList[int(min2 / 20)], ((min2 + 1) % 20))
            else:
                path2 = 'train/{}/image_00{}.jpg'.format(imageList[int(min2 / 20)], (min2 + 1) % 20)

            if (((min3 + 1) % 20) < 10):
                if(((min3 + 1) % 20) == 0):
                    path3 = 'train/{}/image_00{}.jpg'.format(imageList[int(min3 / 20)], ((min3 + 1) % 20) + 20)
                else:
                    path3 = 'train/{}/image_000{}.jpg'.format(imageList[int(min3 / 20)], ((min3 + 1) % 20))
            else:
                path3 = 'train/{}/image_00{}.jpg'.format(imageList[int(min3 / 20)], (min3 + 1) % 20)

            if ((min4 + 1) % 20 < 10):
                if (((min4 + 1) % 20) == 0):
                    path4 = 'train/{}/image_00{}.jpg'.format(imageList[int(min4 / 20)], ((min4 + 1) % 20) + 20)
                else:
                    path4 = 'train/{}/image_000{}.jpg'.format(imageList[int(min4 / 20)], ((min4 + 1) % 20))
            else:
                path4 = 'train/{}/image_00{}.jpg'.format(imageList[int(min4 / 20)], (min4 + 1) % 20)

            if (((min5 + 1) % 20) < 10):
                if (((min5 + 1) % 20) == 0):
                    path5 = 'train/{}/image_00{}.jpg'.format(imageList[int(min5 / 20)], ((min5 + 1) % 20) + 20)
                else:
                    path5 = 'train/{}/image_000{}.jpg'.format(imageList[int(min5 / 20)], ((min5 + 1) % 20))
            else:
                path5 = 'train/{}/image_00{}.jpg'.format(imageList[int(min5 / 20)], (min5 + 1) % 20)

            im_train1 = Image.open(path1)
            im_train2 = Image.open(path2)
            im_train3 = Image.open(path3)
            im_train4 = Image.open(path4)
            im_train5 = Image.open(path5)
            im_test = Image.open(testPath)

            save_test_img_rgb = concatImagesWidth(im_test, im_train1, im_train2, im_train3, im_train4, im_train5)

            min1 = 0
            min2 = 1
            min3 = 2
            min4 = 3
            min5 = 4
            for b in range(5, 120):
                if hue_list[b] < hue_list[min1]:
                    min1 = b
                elif hue_list[b] < hue_list[min2]:
                    min2 = b
                elif hue_list[b] < hue_list[min3]:
                    min3 = b
                elif hue_list[b] < rgb_list[min4]:
                    min4 = b
                elif hue_list[b] < hue_list[min5]:
                    min5 = b

            if imageList[i] == imageList[int(min1/20)] or imageList[i] == imageList[int(min2/20)] or imageList[i] == imageList[int(min3/20)] or imageList[i] == imageList[int(min4/20)] or imageList[i] == imageList[int(min5/20)]:
                successRateHue+= 1

            if (((min1 + 1) % 20) < 10):
                if (((min1 + 1) % 20) == 0):
                    path1 = 'train/{}/image_00{}.jpg'.format(imageList[int(min1 / 20)], ((min1 + 1) % 20) + 20)
                else:
                    path1 = 'train/{}/image_000{}.jpg'.format(imageList[int(min1 / 20)], ((min1 + 1) % 20))
            else:
                path1 = 'train/{}/image_00{}.jpg'.format(imageList[int(min1 / 20)], (min1 + 1) % 20)

            if (((min2 + 1) % 20) < 10):
                if (((min2 + 1) % 20) == 0):
                    path2 = 'train/{}/image_00{}.jpg'.format(imageList[int(min2 / 20)], ((min2 + 1) % 20) + 20)
                else:
                    path2 = 'train/{}/image_000{}.jpg'.format(imageList[int(min2 / 20)], ((min2 + 1) % 20))
            else:
                path2 = 'train/{}/image_00{}.jpg'.format(imageList[int(min2 / 20)], (min2 + 1) % 20)

            if (((min3 + 1) % 20) < 10):
                if (((min3 + 1) % 20) == 0):
                    path3 = 'train/{}/image_00{}.jpg'.format(imageList[int(min3 / 20)], ((min3 + 1) % 20) + 20)
                else:
                    path3 = 'train/{}/image_000{}.jpg'.format(imageList[int(min3 / 20)], ((min3 + 1) % 20))
            else:
                path3 = 'train/{}/image_00{}.jpg'.format(imageList[int(min3 / 20)], (min3 + 1) % 20)

            if ((min4 + 1) % 20 < 10):
                if (((min4 + 1) % 20) == 0):
                    path4 = 'train/{}/image_00{}.jpg'.format(imageList[int(min4 / 20)], ((min4 + 1) % 20) + 20)
                else:
                    path4 = 'train/{}/image_000{}.jpg'.format(imageList[int(min4 / 20)], ((min4 + 1) % 20))
            else:
                path4 = 'train/{}/image_00{}.jpg'.format(imageList[int(min4 / 20)], (min4 + 1) % 20)

            if (((min5 + 1) % 20) < 10):
                if (((min5 + 1) % 20) == 0):
                    path5 = 'train/{}/image_00{}.jpg'.format(imageList[int(min5 / 20)], ((min5 + 1) % 20) + 20)
                else:
                    path5 = 'train/{}/image_000{}.jpg'.format(imageList[int(min5 / 20)], ((min5 + 1) % 20))
            else:
                path5 = 'train/{}/image_00{}.jpg'.format(imageList[int(min5 / 20)], (min5 + 1) % 20)

            im_train1 = Image.open(path1)
            im_train2 = Image.open(path2)
            im_train3 = Image.open(path3)
            im_train4 = Image.open(path4)
            im_train5 = Image.open(path5)
            im_test = Image.open(testPath)



            save_test_img_hue = concatImagesWidth(im_test, im_train1, im_train2, im_train3, im_train4, im_train5)

            save_img_concat = concatImagesHeight(save_test_img_rgb, save_test_img_hue)
            save_img_concat.save(outputPath + '/{}rgb_hue.jpg'.format(ctr))
            ctr += 1


        totalRgbSuccess += successRateRgb
        totalHueSuccess += successRateHue
        print('success rate for {} rgb'.format(imageList[i]) + ' : %' , (successRateRgb/10)* 100)
        print('success rate for {} hue'.format(imageList[i]) + ' : %', (successRateHue / 10) * 100)
    print('\ntotal success rate for rgb : %', (totalRgbSuccess / 6))
    print('\ntotal success rate for hue : %', (totalHueSuccess / 6))
findSimilars()