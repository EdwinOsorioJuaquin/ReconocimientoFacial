# Proyecto Final de Reconocimiento Facial

Implementaremos una red neuronal usando keras-tensorflow y la ejecutaremos en un servicio web de flask

## 1. Preparación del entorno

    $ conda create -n APIRF anaconda python=3.7.16
    $ conda activate APIRF
    $ conda install ipykernel
    $ python -m ipykernel install --user --name APIRF --display-name "APIRF"
    $ conda install tensorflow-gpu==2.6.0 cudatoolkit=11.3.1
    $ pip install tensorflow==2.6.0
    $ pip install jupyter
    $ pip install keras==2.6.0
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]

## 2. Probar el API de Flask

    $ python app.py
