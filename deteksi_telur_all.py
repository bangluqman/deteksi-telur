#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:41:54 2024

@author: hakim
"""

import cv2
import streamlit as st
import numpy as np
from PIL import Image

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright

def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img

def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def main_loop():
    st.title("Deteksi Telur Cacing")
    st.subheader("Aplikasi ini digunakan untuk mendeteksi telur cacing pada feses")
    st.text("Aplikasi ini menggunakan OpenCV")

    gambar = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not gambar:
        return None

    gambar = Image.open(gambar)
    gambar = np.array(gambar)

    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 10000
    largest_ellipses = []

    for contour in contours:
        if len(contour) >= 8:
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            area = np.pi * major_axis * minor_axis / 4

            if area > min_area:
                largest_ellipses.append(ellipse)

    gambar_hasil = gambar.copy()
    for ellipse in largest_ellipses:
        cv2.ellipse(gambar_hasil, ellipse, (0, 255, 0), 2)

    # Display images in Streamlit
    st.text("Original Image vs Processed Images")
    st.image([gambar, edges, gray, gambar_hasil], caption=['Original', 'Edges', 'Grayscale', 'With Ellipses'], use_column_width=True)

if __name__ == '__main__':
    main_loop()
