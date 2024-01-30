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
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")



    gambar = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not gambar:
        return None

    gambar = Image.open(gambar)
    gambar = np.array(gambar)

    # Konversi gambar menjadi grayscale
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi menggunakan metode Canny
    edges = cv2.Canny(gray, 50, 150)

    # Temukan kontur pada gambar yang telah di-deteksi tepinya
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter elips berdasarkan ukuran
    min_area = 10000  # Ganti dengan nilai sesuai dengan ukuran minimum telur cacing yang diinginkan
    largest_ellipse = None
    largest_area = 0

    # Mekanisme Pencarian Contour
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            area = np.pi * major_axis * minor_axis / 4
            
            # Memilih elips terbesar
            if area > min_area and area > largest_area:
                largest_area = area
                largest_ellipse = ellipse
                
                

    # Gambar elips terbesar yang terdeteksi pada gambar asli
    gambar_hasil = gambar.copy()
    if largest_ellipse is not None:
        cv2.ellipse(gambar_hasil, largest_ellipse, (0, 255, 0), 2)

    
    
    st.text("Original Image vs Processed Image")
    # st.image([gambar,edges, gambar_hasil])
    st.image([gambar,edges, gray,gambar_hasil])
    st.text(largest_ellipse)
        



if __name__ == '__main__':
    main_loop()
