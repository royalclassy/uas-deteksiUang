from PIL import Image

image = Image.open('gambar5.jpg').convert('L')

def my_threshold(type):
    width, height = image.size

    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            # Binary Threshold
            if type == 1:
                threshold_value = 128
                if pixel_value > threshold_value:
                    image.putpixel((x, y), 255)
                else:
                    image.putpixel((x, y), 0)
            # Binary Inverted Threshold
            if type == 2:
                higher_threshold = 180
                lower_threshold = 140
                if (pixel_value > lower_threshold and pixel_value < higher_threshold):
                    image.putpixel((x, y), 0)
                else:
                    image.putpixel((x, y), 255)
            # Truncate Threshold
            if type == 3:
                threshold_value = 128
                if pixel_value < threshold_value:
                    image.putpixel((x, y), threshold_value)
            # Truncate To Zero Threshold
            if type == 4:
                threshold_value = 128
                if pixel_value < threshold_value:
                    image.putpixel((x, y), 0)
            # Truncate To Zero Invert Threshold
            if type == 5:
                threshold_value = 128
                if pixel_value > threshold_value:
                    image.putpixel((x, y), 0)

    return image

image.show()
my_threshold(3).show()

# import cv2
# import numpy as np

# # Membaca gambar
# image = cv2.imread('gambar5.jpg')

# # Mengubah gambar menjadi grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Menghitung histogram grayscale
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# # Menentukan nilai intensitas piksel terang terendah (threshold)
# threshold = np.argmax(hist)

# # Mengaplikasikan threshold untuk mengubah bagian gambar yang tidak terlalu gelap menjadi paling gelap
# dark_mask = gray <= threshold
# dark_pixels = np.where(dark_mask)
# gray[dark_pixels] = 0

# cv2.namedWindow('Hasil', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hasil', 400, 600)

# # Menampilkan gambar hasil
# cv2.imshow('Hasil', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
# """
# Created on Fri Apr 14 09:23:22 2023

# @author: Esterlita
# """

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('gambar5.jpg',0)
# laplacian64f = cv.Laplacian(img,cv.CV_64F)
# abs_laplacian64f = np.absolute(laplacian64f)
# laplacian_8u = np.uint8(abs_laplacian64f)
# sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
# abs_sobelx64f = np.absolute(sobelx64f)
# sobelx_8u = np.uint8(abs_sobelx64f)
# sobely64f = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
# abs_sobely64f = np.absolute(sobely64f)
# sobely_8u = np.uint8(abs_sobely64f)
# magnitudesobel = cv.magnitude(sobelx64f,sobely64f)
# abs_sobel64f = np.absolute(magnitudesobel)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(3,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,2,2),plt.imshow(laplacian_8u,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,2,3),plt.imshow(sobelx_8u,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,2,4),plt.imshow(sobely_8u,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,2,5),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel Magnitude'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.imshow('Original', img)
# cv.imshow('Laplacian', laplacian_8u)
# cv.imshow('Sobel X', sobelx_8u)
# cv.imshow('Sobel Y', sobely_8u)
# cv.imshow('Sobel Magnitude', sobel_8u)
# cv.waitKey(0)
# cv.destroyAllWindows()
