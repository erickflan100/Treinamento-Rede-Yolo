!pip install tensorflow
!pip install keras

!pip install -q -U keras

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Baixar o modelo YOLOv3 pré-treinado
!wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights
# Use the 'raw' URL to download the configuration file
!wget -O yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg


def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

net, output_layers = load_yolo_model()
print("Modelo YOLOv3 carregado com sucesso!")

# Baixar o arquivo de classes COCO
!wget -O coco.names https://github.com/pjreddie/darknet/blob/master/data/coco.names

# Carregar os nomes das classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Total de {len(classes)} classes carregadas.")

!ls -lh yolov3.*

!git clone https://github.com/qqwweee/keras-yolo3.git
%cd keras-yolo3

# Criar nova saída para Transfer Learning
def modify_yolo_for_transfer_learning(base_model, num_classes):
    x = base_model.output
    x = Conv2D(1024, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Nova camada de saída com a quantidade de classes personalizadas
    output_layer = Conv2D(num_classes, (1, 1), activation="softmax")(x)

    new_model = Model(inputs=base_model.input, outputs=output_layer)
    return new_model

# Número de classes personalizadas (exemplo: cachorro e gato)
NUM_CLASSES = 2
new_model = modify_yolo_for_transfer_learning(net, NUM_CLASSES)
new_model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurar o gerador de imagens para aumentar os dados
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

# Carregar imagens do dataset personalizado
train_generator = train_datagen.flow_from_directory(
    'caminho_para_seu_dataset/train',  # Substitua pelo caminho do seu dataset
    target_size=(416, 416),
    batch_size=8,
    class_mode='categorical')

# Compilar o modelo
new_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Treinar o modelo
new_model.fit(train_generator, epochs=10)

!git clone <https://github.com/AlexeyAB/darknet>
%cd darknet
!make

# Baixar os pesos pré-treinados do YOLOv4
!wget <https://pjreddie.com/media/files/yolov4.weights>

# Criar estrutura de diretórios para dataset
import os
os.makedirs("data/images/train", exist_ok=True)
os.makedirs("data/images/test", exist_ok=True)
os.makedirs("data/labels/train", exist_ok=True)
os.makedirs("data/labels/test", exist_ok=True)

# Exemplo de como mover suas imagens para o diretório correto
# Se estiver usando COCO, baixe os arquivos aqui
!wget <http://images.cocodataset.org/zips/train2017.zip>
!unzip train2017.zip -d data/images/train

# Criar arquivos de configuração do YOLO
with open("data/obj.names", "w") as f:
    f.write("classe1\\n")  # Substitua pelo nome da sua classe
    f.write("classe2\\n")

with open("data/obj.data", "w") as f:
    f.write("classes = 2\\n")
    f.write("train = data/train.txt\\n")
    f.write("valid = data/test.txt\\n")
    f.write("names = data/obj.names\\n")
    f.write("backup = backup/\\n")

# Criar um arquivo de configuração baseado no yolo.cfg
!cp cfg/yolov4.cfg cfg/yolo-obj.cfg
!sed -i 's/batch=64/batch=16/' cfg/yolo-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=8/' cfg/yolo-obj.cfg

# Criar lista de imagens para treinamento e validação
import glob
train_images = glob.glob("data/images/train/*.jpg")
with open("data/train.txt", "w") as f:
    f.write("\\n".join(train_images))

test_images = glob.glob("data/images/test/*.jpg")
with open("data/test.txt", "w") as f:
    f.write("\\n".join(test_images))

# Treinamento do YOLOv4
!./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.weights -dont_show -map

# Testando o modelo treinado
!./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights data/images/test/exemplo.jpg