import streamlit as st
import pandas as pd
import numpy
import cv2
import numpy as np
from PIL import Image
import joblib
import pywt
import os

st.set_page_config(
        page_title="Politican Face Recognition",
)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: black;
    color : white;
}
img {
  border-radius : 50%;
  width:120px;
  height:120px;
}

</style>""", unsafe_allow_html=True)


face_cascade = cv2.CascadeClassifier( './model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
model_lr = joblib.load("./model/best_model_lr.joblib")
model_svm = joblib.load("./model/best_model_lr.joblib")
class_dict = joblib.load("./model/class_dict.joblib")



# wavelet transform is used to find features in signal using decomposition and then reconstructing the signal
def w2d(img,mode = "haar",level = 1):
    imgArray = img
    #convert to grayscale
    imgArray = cv2.cvtColor(imgArray,cv2.COLOR_BGR2GRAY)
    #convert to float
    imgArray = imgArray.astype(np.float64)
    #scaling the image
    imgArray = imgArray / 255
    #decomposing the signal
    coeffs = pywt.wavedec2(imgArray,mode,level = level)
    #print(coeffs)
    #process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    #reconstructing the signal back in original form
    imgArray_H = pywt.waverec2(coeffs_H,mode)
    imgArray_H *= 255
    imgArray_H = imgArray_H.astype(np.uint8)
    return imgArray_H

def get_cropped_face(img,send_two = "no"):
    #print(img_path)
    #img = cv2.imread(img_path)
    img = Image.open(img)
    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if img is not None:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,3)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                if send_two == "yes":
                    for (x,y,w,h) in faces:
                        face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_color = face_img[y:y+h,x:x+w]
                        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
                    #print(len(eyes))
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                    return face_img
                return roi_color



def load_predict_image(image):
    X = []
    img = get_cropped_face(image)
    image_file = Image.open(image)
    img_array = np.array(image_file)
    img_array = cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR)
    image_file = cv2.resize(img_array,(256,256))
    #img = get_cropped_face(image_file)
    #cv2_image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if img is not None:
        scaled_raw_image = cv2.resize(img,(32,32))
        img_har = w2d(img,"db1",5)
        scaled_img_har = cv2.resize(img_har,(32,32))
        # converting colored image in 2D array of (32*32*3,1) here 3 are channels BGR
        scaled_raw_image_2D = scaled_raw_image.reshape((32*32*3,1))
        # converting feature(black and white) image in 2D array of (32*32,1)
        scaled_img_har_2D = scaled_img_har.reshape((32*32,1))
     
        combined_img = np.vstack((scaled_raw_image_2D,scaled_img_har_2D))
        X.append(combined_img)
        #st.write(len(combined_img))
        X = np.array(X).reshape((len(X),4096)).astype(float)
        # this saved model has model pipeline(StandardScaler(),svm()) in it
        #st.write(model_lr.predict(X))
        #st.write(model_svm.predict(X))
        svm_prediction = model_svm.predict(X)
        #lr_prediction = model_lr.predict(X)
        svm_proba = model_svm.predict_proba(X)
        #lr_proba = model_lr.predict_proba(X)
        #st.text(svm_proba)
        #st.text(lr_proba)
        svm_max_index = np.argmax(svm_proba)
        #lr_max_index = np.argmax(lr_proba)
        #st.text(svm_prediction[0])
        #st.text(lr_prediction[0])
        #st.text(svm_max_index)
        #st.text(svm_proba[0][svm_max_index])
        #st.text(lr_proba[0][lr_max_index])
        key_list = list(class_dict.keys())
        values_list = list(class_dict.values()) 
        #st.write(class_dict)
        df = pd.DataFrame(key_list,columns = ["Celebrity"])
        df["Probability_score"] = svm_proba[0]
        svm_position = values_list.index(svm_prediction[0])
        #lr_position = values_list.index(lr_prediction[0])
        svm_result = key_list[svm_position]
        if not svm_proba[0][svm_max_index] >= 0.6:
            svm_result = "Sorry! No Match Found"
        #lr_result = key_list[lr_position]
        #if not lr_proba[0][lr_max_index] > 0.5:
            #lr_result = "Sorry! This Image cannot be predicted"
        #X.append(combined_img)
        #y.append(class_dict[poli_name])
    else:
        key_list = list(class_dict.keys())
        values_list = list(class_dict.values())
        df = pd.DataFrame(key_list,columns = ["Celebrity"])
        df["Probability_score"] = np.zeros(4)
        svm_result = "Sorry! No Face Detected"
    return image_file,svm_result,df



def main():
    st.title("Politician Face Recognition")
    menu = ["Home","Model Building Stages"]
    choice = st.sidebar.radio('Menu',menu)

    if choice == "Home":
        st.subheader("Home")
        col5,col6,col7,col8 = st.columns(4)
        front_image_path = "./front_images/"
        count = 5
        columns_dict = {
            '5':col5,
            '6':col6,
            '7':col7,
            '8':col8,


        } 
        
        #st.write(columns_dict)
        for image_path in os.scandir(front_image_path):
            #st.write(image_path.path)
            politician_image = Image.open(image_path.path)
            image_caption = image_path.path.split('/')
            #st.write(image_caption)
            image_caption = image_caption[2].split('.')[0] 
            columns_dict[f'{count}'].image(politician_image,caption = f"{image_caption}")
            count +=1

        image = st.file_uploader("Upload Image",["png","jpg","jpeg"],accept_multiple_files = False)
        if st.button('Predict'):
            if image is not None:
                #st.write(type(image))
                #attributes and methods
                #st.write(dir(image))
                file_details = {
                    "name": image.name,
                    "type": image.type,
                    "size":image.size,
                }
                #st.write(file_details)
                #if you need to pass it to a open cv 
                img_array = np.array(image)
                # image_opencv = cv2.im
                col1,col2 = st.columns(2)
                image_loaded,svm_result,df = load_predict_image(image)
                col1.header("Results")
                col1.image(image_loaded,caption = f"{svm_result}",channels = 'BGR')
                col2.header("Probability Score")
                col2.write(df)
                #st.text(svm_result)
                #st.text(lr_result)
    else:
        m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: black;
            color : white;
        }
        img {
        border-radius: 50px;
        width:160px;
        height:160px;
        }

        </style>""", unsafe_allow_html=True)
        st.text("Source code : https://github.com/saqibzia-dev/app-bengaluru-housing-price-prediction")
        st.subheader("Model Building Stages:")
        
        dataset = st.container()
        face_detection = st.container()
        dataset = st.container()
        model_selection = st.container()
        final_model_results = st.container()
            
        

        with dataset:
            st.subheader("1.Data Collection:")
            st.text("Images were collected from google.com using fatkun extension ")
            #dataset = pd.read_csv("data/bengaluru_house_prices.csv")
            #st.write(dataset.head())
        with face_detection:
            st.subheader("2.Feature Detection:")
            #st.text("Face and Eyes are detected using open cv haarcascades ")
            
            get_image = './model/dataset/front_images/shahbaz.jpg'
            original_image = Image.open(get_image)
            cropped_face = get_cropped_face(get_image,send_two = "yes")
            

            #plt.imshow(cropped_face)
            st.markdown("* Face and Eyes are detected using open cv haarcascades ")
            st.markdown("* Use above filters we crop our face ")
            st.markdown("* Features  are detected using  wavelet transform")
            col1,col2,col3 = st.columns(3)
            col1.image(original_image,caption = "Original")
            col2.image(cropped_face,channels = "BGR",caption = "haarcascades detection")
            cropped_face = get_cropped_face(get_image)
            cropped_image = np.array(cropped_face)
            #print(cropped_image.shape)
            im_har = w2d(cropped_image,'db1',5)
            col3.image(im_har,caption = "wavelet transform")
            st.text("These steps will be repeated on all images to get features")


        with dataset:
            st.header("3. Dataset Creation:")
            st.markdown("* **Image resize:** Each cropped image is resized into 32x32 size")
            st.markdown("* **Wavelet transform:** On above images we apply wavelet transform")
            st.markdown("* **reshaping cropped image:** Then we reshape each cropped image into (32*32*3,1) ")
            st.markdown("* **reshaping wavelet transformed image:** Then we reshape each wavelet transformed image into (32*32,1) ")
            st.markdown("* **Combining:** Then both of these images are combined using  vertical stacking ")
            image_dataset = pd.read_csv('./model/dataset.csv')
            st.write(image_dataset.head(5))


        with model_selection:
            
            st.header("4.Selecting Best Model")
            st.text("Using gridsearchcv to find best model and best hyper parameters")
            st.subheader("Here are the results")
            model_scor = pd.read_csv('./model/GridSearchCV_results.csv')
            st.write(model_scor)
            st.text("Here I am selecting svm" )

        with final_model_results:
            st.header("5.Test set results:")
            st.markdown("* **SVC accuracy :** 0.88 ")
            st.markdown("* **Logistic Regression accuracy:** 0.91 ")
            




if __name__ == '__main__':
    main() 
