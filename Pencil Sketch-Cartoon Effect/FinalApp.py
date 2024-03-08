import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to convert image to pencil sketch
def pencil_sketch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final = cv2.divide(img_gray, 255 - img_smoothing, scale=256)
    return final

# Function to convert image to cartoon
def cartoonize(img):
    line_size, blur_value = 5, 7
    edges = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_size, blur_value)
    img_quantized = color_quantization(img, k=5)
    blurred = cv2.bilateralFilter(img_quantized, d=3, sigmaColor=200, sigmaSpace=200)
    cartoon_image = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon_image

# Function for color quantization
def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Streamlit app
def main():
    # Add logo
    logo = Image.open("logo.jpg")
    st.image(logo, use_column_width=False)
    st.title("IMAGE TO PENCIL SKETCH AND CARTOON CONVERTER")
    # Add CSS styling for the logo
    st.markdown(
        """
        <style>
            .stImage > img {
                width: 20px; /* Adjust the width as per your requirement */
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar for file upload and conversion option
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    conversion_option = st.radio("Select Conversion Option:", ("Pencil Sketch", "Cartoon"))

    if uploaded_file is not None:
        # Load and display original image
        img = Image.open(uploaded_file)

        # Convert and display the selected effect
        if conversion_option == "Pencil Sketch":
            st.subheader("Pencil Sketch Effect")
            sketch_img = pencil_sketch(np.array(img))
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            with col2:
                st.image(sketch_img, caption="Pencil Sketch", use_column_width=True)
        elif conversion_option == "Cartoon":
            st.subheader("Cartoon Effect")
            cartoon_img = cartoonize(np.array(img))
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            with col2:
                st.image(cartoon_img, caption="Cartoon Image", use_column_width=True)

        # Save the converted image
        if st.button("Save Image"):
            if conversion_option == "Pencil Sketch":
                Image.fromarray(sketch_img).save("pencil_sketch_output.jpg")
            elif conversion_option == "Cartoon":
                Image.fromarray(cartoon_img).save("cartoon_output.jpg")
            st.success("Image saved successfully!")

if __name__ == "__main__":
    main()
