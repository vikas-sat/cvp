import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf



# Load your trained model
model = load_model("Model.h5")

st.title("Brain Tumor Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan", type=["jpg", "png", "jpeg"])


def highlight_tumor_region(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)

    # Apply Gaussian blur for smoothness
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours from the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an RGB version of the gray image
    colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw the contours on the RGB image
    cv2.drawContours(colored_img, contours, -1, (255, 0, 0), 2)

    # Apply a heatmap for pseudo coloring
    heatmap_img = cv2.applyColorMap(colored_img, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlayed_img = cv2.addWeighted(heatmap_img, 0.5, colored_img, 0.5, 0)

    return overlayed_img

def predict(image_path):
    img = image.load_img(image_path, target_size=(150, 150), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    prediction = model.predict(img)
    return prediction[0][0]


if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classification: ")

    # Save the uploaded file to a directory
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make a prediction
    result = predict("temp_image.jpg")

    # Show the prediction
    if result < 0.35:
        st.write("The MRI scan likely shows a brain tumor.")

        # Show region button
        if st.button("Show Region"):
            highlighted_img = highlight_tumor_region("temp_image.jpg")
            st.image(highlighted_img, caption='Highlighted Tumor Region', use_column_width=True)
    else:
        st.write("The MRI scan likely does not show a brain tumor.")
