import streamlit as st
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras
import random

labels={
    0:'Animal',
    1: 'Cartoon',
    2: 'Chevron',
    3: 'Floral',
    4: 'Geometry',
    5: 'Houndstooth',
    6: 'Ikat',
    7: 'Letternumb',
    8: 'Others',
    9: 'Plain',
}

st.title('Content Recommendation System')
# load the DL Model
model=tensorflow.keras.models.load_model('keras_model.h5')

def load_image(image_file):
    img=Image.open(image_file)
    return img

st.subheader("Enter Input Image")
src_image_file=st.file_uploader("Upload Images",type=["png","jpg","jpeg"])

if src_image_file is not None:
    file_details={"filename":src_image_file,"filetype":src_image_file.type,"filesize":src_image_file.size}
    st.write(file_details)
    st.image(load_image(src_image_file),width=250)

    with open(os.path.join("uploads","src.jpg"),"wb") as f:
        f.write(src_image_file.getbuffer())
        st.success("File Saved")

    data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
    #open the image
    image=Image.open('uploads/src.jpg') # test.jpg
    # resize the image
    size=(224,224)
    image=ImageOps.fit(image,size,Image.ANTIALIAS)
    # convert this into numpy array
    image_array=np.asarray(image)
    # Normalise the Image - (0 to 255)
    normalise_image_array=(image_array.astype(np.float32)/127.0)-1
    # loading the image into the array
    data[0]=normalise_image_array
    # pass this data to model
    prediction=model.predict(data)
    print(prediction) # [[0.5,0.5,0.7,0.3]]
    # Decision Logic
    prediction=list(prediction[0])
    max_prediction=max(prediction)
    index_max=prediction.index(max_prediction)
    print(index_max)
    #st.text("Expected Result: "+labels[index_max])
    temp_paths=[]
    temp_path=os.path.join('data',labels[index_max].upper())
    #st.write(temp_path)
    temp_paths=(os.listdir(temp_path))
    random_temp_paths=[]
    for i in range(5):
        k=random.randrange(len(temp_paths))
        random_temp_paths.append(temp_paths[k])
    st.header('Recommended Dresses')
    #st.write(random_temp_paths)
    col1,col2=st.columns(2)
    for i in range(5):
        if i%2==0:
            col1.image(temp_path+'\\'+random_temp_paths[i],width=300)
        else:
            col2.image(temp_path+'\\'+random_temp_paths[i],width=300)
        




    

