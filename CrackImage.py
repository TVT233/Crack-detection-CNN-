#-*- coding: UTF-8 -*-
# based on tensorflow==1.2.0 keras==2.0.6
# Also install h5py, opencv-python
import os
# Run on CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
'''ZZH 2022.04.20
   1.将网络结构修改为mobilenetv1'''
'''ZZH 2022.04.28
   将网络模型进行剪枝，减少参数量，成功
   总参数大约12000，卷积层参数减小至原先1/5
   结合之前CNN网络进行修改'''
'''ZZH 2022.04.28
    尝试用卷积层代替全连接层，减小参数至4958'''



import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import numpy as np
import cv2
import random
# from sklearn.metrics  import roc_curve, auc

# Solve "Function call stack:train_function"
# tf.compat.v1.disable_eager_execution()

# 动态分配GPU资源
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


BATCH_SIZE = 64
epochs = 8
Image_resize = (99, 99)
Input_shape = (99, 99, 3)  #要求输入的大小 
Dropoutrate = 0.5



'''def conv_block(input_tensor, filters, alpha, kernel_size=(9,9), strides=(1,1)):
 
    # 超参数alpha控制卷积核个数
    filters = int(filters*alpha)
 
    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size, 
                      strides=strides,  # 步长
                      padding='same',   # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置
 
    x = layers.BatchNormalization()(x)  # 批标准化
 
     #x = layers.Activation("relu")(x)  # relu6激活函数
    x = layers.Activation('relu')(x)
 
    return x  # 返回一次标准卷积后的结果'''
 
#（2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters,alpha, depth_multiplier,kernel_size=(3,3),  strides=(1,1)):
 
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters*alpha)
 
    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,  # 卷积核size
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置
 
    x = layers.BatchNormalization()(x)  # 批标准化
 
    x = layers.Activation('relu')(x) # relu6激活函数
 
    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1,1),  # 卷积核默认1*1
                      padding='same',   # 卷积过程中特征图size不变
                      strides=(1,1),   # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置
 
    x = layers.BatchNormalization()(x)  # 批标准化
 
    x = layers.Activation('relu')(x) # 激活函数
 
    return x  # 返回深度可分离卷积结果




'''zm change 2021.4.26 模型结构化简版'''
# V3.0 try AlexNet
def make_model_MobileNetV1(classes, input_shape, alpha, depth_multiplier, dropout_rate):
    inputs = Input(shape=input_shape)
    #inputs = np.expand_dims(inputs, axis=3)

    #Parameter meaning: number of convolutional kernels, convolutional kernel size, step size and padding mode, 
    #same means padding with 0 for edges, valid means no padding
    x = depthwise_conv_block(inputs,32, alpha,  depth_multiplier,kernel_size=(8,8), strides=(2,2))#50*50*32
    x = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)#24*24*32
    x = layers.Dropout(rate=Dropoutrate)(x)

    x = depthwise_conv_block(x,16, alpha,  depth_multiplier,kernel_size=(5,5), strides=(1,1))#24*24*16
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)#11*11*16
    x = layers.Dropout(rate=Dropoutrate)(x)

    '''x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)'''

    # Add a new conv layer
    '''x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)'''

    x = depthwise_conv_block(x,128, alpha,  depth_multiplier,kernel_size=(3,3), strides=(1,1))#11*11*128
    #x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)#5*5*8
    #x = layers.Dropout(rate=Dropoutrate)(x)

    # x = layers.Flatten()(x)
    # x = layers.Dense(48)(x)
    # x = layers.Activation("relu")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(rate=Dropoutrate)(x)
    '''1.全连接层
    x = layers.Dense(96)(x)
    # x = layers.Activation("relu")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(rate=Dropoutrate)(x)
    '''


    # # always softmax
    # if classes == 1111:
    #     activation = "sigmoid"
    #     units = 1
    # else:
    #     activation = "softmax"
    #     units = classes

    #x = layers.Dense(units)(x)
    #outputs = layers.Activation(activation=activation)(x)
    #outputs = layers.Dense(units=units, activation=activation)(x)
    x = layers.GlobalAveragePooling2D()(x)  # 通道维度上对size维度求平均
 
    # 超参数调整卷积核（特征图）个数
    shape = (1, 1, int(128 * alpha))
    #shape = (1, 1, 1024)
    # 调整输出特征图x的特征图个数
    x = layers.Reshape(target_shape=shape)(x)
 
    # Dropout层随机杀死神经元，防止过拟合
    x = layers.Dropout(rate=dropout_rate)(x)
 
    # 卷积层，将特征图x的个数转换成分类数
    x = layers.Conv2D(classes, kernel_size=(1,1), padding='same')(x)
 
    # 经过softmax函数，变成分类概率
    x = layers.Activation('softmax')(x)
 
    # 重塑概率数排列形式
    x = layers.Reshape(target_shape=(classes,))(x)


    print(type(inputs))
    print(type(x))
    outputs = Model(inputs, x)
    return outputs


# Read IMAGE file Path and Label 从文件里读取图片并与标签对应，相当于构建数据集
# normal--0, crack--1
def IMG_Lst(dataset_root):
    images_list = np.array([])
    labels_list = np.array([])
    '''zm change 2021.4.26 修改输入的路径'''
    # 制作网络模型所需数据集即让图片和对应的标签信息对应，normal为正常图片，crack为含裂缝图像 
    for root, dirs, files in os.walk(dataset_root + '/normal'):
        for file in files:
            images_list = np.append(images_list, root + '/' + file)
            labels_list = np.append(labels_list,[0])
    for root, dirs, files in os.walk(dataset_root + '/crack'):
        for file in files:
            images_list = np.append(images_list, root + '/' + file)
            labels_list = np.append(labels_list,[1])
    return images_list, labels_list

def IMG_RD(images_list, label_list, Image_resize):
    img_lst = []
    for i in range(len(images_list)):
        # cv2.imread(文件名，标记)读入图像 1表示彩色图像 默认为BGR
        img_tmp = cv2.imread(images_list.copy()[i], 1)#0表示灰度图像 1表示BGR
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)#BGR=>RGB
        #print("---------------------------------------")
       # print(img_tmp.shape)
        img_tmp_rsz = cv2.resize(img_tmp, Image_resize, interpolation=cv2.INTER_CUBIC)
       # print("---------------------------------------")
       # print(img_tmp_rsz.shape)
       # print("---------------------------------------")
        #img_tmp_rsz = np.expand_dims(img_tmp_rsz, axis=3)

        img_lst.append(img_tmp_rsz)
    return np.array(img_lst), label_list



if __name__ == '__main__':
    # *******************************************************************
    # Data loading and preprocessing

    
    # 读取训练数据集及测试/验证数据集的RGB图像，将所有图像resize为image_size的大小
    images_list, labels_list = IMG_Lst('C:/Users/13291/Desktop/dcds/CHECK')#Concrete Crack Images for Classification
    Trainset_image, Trainset_label = IMG_RD(images_list, labels_list, Image_resize)
    Trainset_label_one_hot = tf.keras.utils.to_categorical(Trainset_label, num_classes=2)
    print('Read Images Done!')
    print(np.size(Trainset_image))
    # 随机打乱图像信息
    c = list(zip(Trainset_image, Trainset_label, Trainset_label_one_hot))  # 将a,b整体作为一个zip,每个元素一一对应后打乱
    c = [i for i in c]
    random.shuffle(c)  # 打乱c
    Trainset_image[:], Trainset_label[:], Trainset_label_one_hot[:] = zip(*c)  # 将打乱的c解开'''

    # *******************************************************************
    # Building Models with Keras
    # input_shape=(rows, cols, channels)  data_format='channels_last'(default).
    model = make_model_MobileNetV1(classes=2,  # 分类种类数
                      input_shape=[99,99,3],   # 模型输入图像shape
                      alpha=1.0,  # 超参数，控制卷积核个数
                      depth_multiplier=1,  # 超参数，控制图像分辨率
                      dropout_rate=1e-3)  # 随即杀死神经元的概率)
    model.summary()

    # ******************************************************************
    # 模型编译及训练
    model.compile(
        optimizer="adam",  # keras.optimizers.Adam(),
        loss="categorical_crossentropy",  # keras.losses.mean_squared_error, # "categorical_crossentropy",
        metrics=['accuracy', keras.metrics.categorical_accuracy]  # [keras.metrics.binary_accuracy]  # ["accuracy"]
    )
    
    filepath="C:/Users/13291/Desktop/dcds/1.03_weights.best.h5"
    #保存最佳模型
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    history = model.fit(
        Trainset_image, Trainset_label_one_hot,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        validation_split = 0.2, #从测试集中划分80%给训练集
        #validation_freq = 1, #测试的间隔次数为1
        #validation_data = (Trainset_image * 1.0 / 255, Trainset_label_one_hot),
        # callbacks=callbacks,
    )

    model.save('C:/Users/13291/Desktop/dcds/1.03_CrackImage_v1.00_Model_b%d_e%d_d%f.h5' % (BATCH_SIZE, epochs, Dropoutrate))
    
    # ******************************************************************
    #验证集准确度曲线图
    val_accuracy = history.history['val_accuracy']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Validation Accuracy')
    plt.savefig('C:/Users/13291/Desktop/dcds/1.03_val_accuracy_b%d_e%d_d%f_1.jpg' % (BATCH_SIZE, epochs, Dropoutrate))
    plt.show()
    
    #验证集loss曲线图
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.title('Validation Loss')
    plt.savefig('C:/Users/13291/Desktop/dcds/1.03_Validation_loss_b%d_e%d_d%f.jpg' % (BATCH_SIZE, epochs, Dropoutrate))
    plt.show()
    
    
    # ******************************************************************a
    # 测试数据集的验证
    # 读取训练数据集及测试/验证数据集的RGB图像，将所有图像resize为image_size的大小
    # images_list, labels_list = IMG_Lst('Concrete Crack Images for Classification')
    # Testset_image, Testset_label = IMG_RD(images_list, labels_list, Image_resize)
    # Testset_label_one_hot = keras.utils.to_categorical(Testset_label, num_classes=2)
    # print('Read Images Done!')
    #
    # # load model
    # model = keras.models.load_model('1.03_weights.best.h5')
    # loss_and_metrics = model.evaluate(Testset_image, Testset_label_one_hot)
    # print(loss_and_metrics)
    # Testset_pred = model.predict(Testset_image, batch_size=BATCH_SIZE)
    # Testset_pred_label = np.argmax(Testset_pred, axis=1)
    # print("Test Accuracy = ", (Testset_pred_label == Testset_label).mean() * 100)
    #Trainset_pred = model.predict(Trainset_image, batch_size=BATCH_SIZE)
    #Trainset_pred_label = np.argmax(Trainset_pred, axis=1)
    #print("Train Accuracy = ", (Trainset_pred_label == Trainset_label).mean() * 100)

    # 绘制ROC 计算AUC # 在windows下完成
    
    






