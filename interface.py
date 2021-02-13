import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def prepare_image(img):
    img_size = 64
    image = img.resize((img_size, img_size))
    image = np.array(image).reshape(1, 64, 64, 3)
    return image

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.write('''<div style="text-align:center; 
                            background_color:#23688f;
                            color:white;
                            font-weight:50%;
                            font-family:"Gill Sans EXtrabold", Helvetica, sans-serif;"><h2>
    Clasificación de Números realizados con la Mano</h2></div>''', unsafe_allow_html=True)

    menu = ["Sobre la App", "Cargar Imagen", "Cámara"]
    st.sidebar.write('''<div style="color:#23688f"><h3>Bienvenido</h3></div>''', unsafe_allow_html=True)
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    col1, col2 = st.beta_columns(2)

    if choice == "Cargar Imagen":
        col1.write('''<div style="color:#23688f;
                                margin-left:5%;">
                                <h3>Aquí se muestra la Imagen</h3>
                    </div>''', unsafe_allow_html=True)

        #st.sidebar.title('Cargar Imagen')
        image_file = st.sidebar.file_uploader("", type=['jpg', 'jpeg'])

        if image_file is not None:
            #st.write(type(image_file))
            # st.write(dir(image_file))
            file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
            #st.write(file_details)

            img = load_image(image_file)
            col1.image(img, width=250)

            data = prepare_image(img)

            col2.write('''<div style="margin-top:25%;"> </div>''', unsafe_allow_html=True)

            if col2.button('Clasificar', key="1"):
                prediction = model.predict([data])
                #print(prediction)
                col2.write("El número en la imagen corresponde a {}".format((prediction[0].tolist()).index(max(prediction[0].tolist()))))
    elif choice == "Cámara":
        st.sidebar.write("")
        col1.write('''<div style="color:#23688f;
                                margin-left:5%;">
                                <h3>Cámara Activa</h3>
                    </div>''', unsafe_allow_html=True)
        FRAME_WINDOW = col1.image([])
        camera = cv2.VideoCapture(0) # 0 corresponde al índice de la cámara

        col2.write('''<div style="margin-top:25%;"> </div>''', unsafe_allow_html=True)
        capture = col2.button('Tomar Foto', key="2")
        col2.write("")
        
        while choice == 'Cámara':
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            if capture:
                image = cv2.resize(frame, (64, 64))
                data = image.reshape(1, 64, 64, 3)
                prediction = model.predict([data])
                #print(prediction)
                col2.write("El número en la imagen corresponde a {}".format((prediction[0].tolist()).index(max(prediction[0].tolist()))))
                capture = False
    else:
        col1.write('''<div style="color:#23688f;
                                margin-left:5%;">
                                <h3>Sobre la App</h3>
                    </div>
                    <p style="border: 5px solid #23688f; 
                            padding: 1%;
                            border-radius:5px;">
                        Esta es una aplicación de Computer Vision que tiene como objetivo la clasificación de
                        imágenes de números realizados con la mano, desde cero (0) hasta cinco (5).
                    </p>''',unsafe_allow_html=True)
        col2.write('''<div style="margin-top:15%;
                                border: 5px solid #23688f; 
                                padding: 1%;
                                border-radius:5px;">
                                <h4 style="color:black;">Cargar Imagen</h4>
                    <p>
                        En esta opción, se carga la imagen que se desea clasificar; luego, se presiona
                        el botón de clasificar y la aplicación imprime el resultado.
                    </p>
                    <h4 style="color:black;">Cámara</h4>
                    <p>
                        En esta opción, se toma la captura de la imagen que se desea clasificar
                        y la aplicación imprime el resultado.
                    </p></div>''', unsafe_allow_html=True)
if __name__ == '__main__':
    model = load_model('Network.model')
    main()
