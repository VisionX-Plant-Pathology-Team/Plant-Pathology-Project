import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import random
import cv2
from tensorflow.keras.models import load_model
# from torchvision.transforms import ToTensor, ToPILImage
import tensorflow as tf
import keras
from keras.applications.vgg16 import preprocess_input


st.markdown("## Plant Pathology Demo")
st.markdown(
    """ Let's find your plant's disease."""
)

@st.cache
def create_aug_images(image, num_imgs):
    image = np.array(image)
    doc_aug = A.Compose([
    A.Transpose(),  
    A.Rotate(limit=int(np.random.uniform(0,360)), p=np.round(np.random.uniform(0,1),2)),
    A.HorizontalFlip(p=np.round(np.random.uniform(0,1),2)),  
    A.VerticalFlip(p=np.round(np.random.uniform(0,1),2)),  
    A.GaussNoise(p=np.round(np.random.uniform(0,1),2)),
    A.RandomBrightnessContrast(p=np.round(np.random.uniform(0,1),2)),
    ])
    random.seed(42)
    imgs = []
    captions = []
    for j,i in enumerate(range(num_imgs)):
        augmented = doc_aug(image=image)
        aug_image = augmented["image"]
        imgs.append(aug_image)
        captions.append("Augmented Image " + str(j+1))
        
    return {"images":imgs,
            "captions" : captions}

@st.cache
def read_model_weights():
    resnet_m = load_model("resnet_son.h5")
    densenet_m = load_model("densenet_son.h5")
    return resnet_m, densenet_m


def get_model_selections(col2, col4):
    x = col2.checkbox("ResNet",value=False)
    y = col4.checkbox("DenseNet",value=False)

    return dict({"resnet" : x,
                 "densenet" : y})

def model_predictions(a, data, model_names):
    
    x = np.zeros((1,4))
    num_preds = 0

    for i, (key, value) in enumerate(a.items()):
        if value == True:
            if key == "resnet":
                x += 3*model_names[i].predict(data)
                num_preds += 3
            elif key == "densenet":
                x += 2*model_names[i].predict(data)
                num_preds += 2
            

    result = x/num_preds
    return result

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image1 = Image.open(img_file_buffer)
    st.image(
    [image1], caption=[f"Test image"], use_column_width=True
    )
    st.markdown("# 5 Different Augmentation of Test Image")
    x = create_aug_images(np.array(image1), 5)
    st.image(
    x["images"][:3], caption=x["captions"][:3], width=219
    )
    st.image(
    x["images"][3:], caption=x["captions"][3:], width=219
    )

    st.markdown("# Model Selection")
    col1, col123, col2 = st.beta_columns(3)
    resnet = np.array(Image.open("resnet1.jpg"))
    col1.image(resnet, width=400)
    col2.write("")
    col2.write("")
    col3, col321, col4 = st.beta_columns(3)
    densenet = np.array(Image.open("densenet.jpg"))
    col3.image(densenet, width=400 )
    col4.write("")
    col4.write("")

    model_selection = get_model_selections(col2, col4)

pred_button = st.sidebar.button("Predict")

if pred_button:
    if sum(model_selection.values()) > 0:
        train_img = tf.expand_dims(preprocess_input(np.array(image1.resize((224,224)))),axis=0)
        resnet_m, densenet_m = read_model_weights()
        model_names = [resnet_m, densenet_m]
        predictions = model_predictions(model_selection, train_img, model_names)
        # st.sidebar.write(predictions)
        idx = ["healthy","multiple disease","rust","scab"]
        st.sidebar.dataframe(pd.DataFrame(predictions[0], columns=["Probabilities"], index= idx))
        st.sidebar.write("")
        st.sidebar.write("Predicted Class is: ")
        st.sidebar.markdown("### {}".format(idx[np.argmax(predictions[0])].capitalize()))
        

        pred_button = False

    else:
        st.sidebar.markdown("## Please select at least 1 algorithm.")
        pred_button = False