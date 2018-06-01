from keras.layers import Input, Dense, Conv2D, Reshape, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras import optimizers
import numpy as np
from sklearn import preprocessing

from keras.preprocessing.image import ImageDataGenerator,load_img, array_to_img, img_to_array

class FeatureExtractor:
    def __init__(self):
        self.encoder_model = None
        self.model = None
        return
    
    def build(self, input_dims, opt):
        input_layer = Input(shape=input_dims)
        
        a_one = Conv2D(64, (3,3), activation='relu', padding='same') (input_layer)
        #a_two = BatchNormalization() (a_one)
        a_three = Conv2D(64, (3,3), activation='relu', padding='same') (a_one)
        #a_four = BatchNormalization() (a_three)
        a_five = MaxPooling2D() (a_three)
        block_one = Dropout(0.25) (a_five)
        #block_one = a_five
        
        b_one = Conv2D(128, (3,3), activation='relu', padding='same') (block_one)
        #b_two = BatchNormalization() (b_one)
        b_three = Conv2D(128, (3,3), activation='relu', padding='same') (b_one)
        #b_four = BatchNormalization() (b_two)
        b_five = MaxPooling2D() (b_three)
        block_two = Dropout(0.25) (b_five)
        
        c_one = Conv2D(256, (3,3), activation='relu', padding='same') (block_two)
        #c_two = BatchNormalization() (c_one)
        c_three = Conv2D(256, (3,3), activation='relu', padding='same') (c_one)
        #c_four = BatchNormalization() (c_three)
        c_five = MaxPooling2D() (c_three)
        block_three = Dropout(0.5) (c_five)
        
        d_one = Conv2D(512, (3,3), activation='relu', padding='same') (block_three)
        #d_two = BatchNormalization() (d_one)
        d_three = Conv2D(512, (1,1), activation='relu', padding='same') (d_one)
        #d_four = BatchNormalization() (d_three)
        d_five = MaxPooling2D() (d_three)
        block_four = Dropout(0.2) (d_five)
        
        flat = Flatten() (block_four)
        fc_one = Dense(4096, activation='relu') (flat)
        #block_five = BatchNormalization() (fc_one)
        
        fc_two = Dense(4096, activation='relu') (fc_one)
        #block_six = BatchNormalization() (fc_two)
        
        final = Dense(2, activation='softmax') (fc_two)
        
        self.model = Model(input_layer, final)
        self.feature_extractor = Model(input_layer, flat)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return
    
    def load(self, model_file):
       # self.encoder_model = load_model(encoder_model_file)
        self.model = load_model(model_file)
        return
    
    def train(self, train_generator,
             validation_generator,
             epochs=50,
             batch_size=1,
             shuffle=True):
        tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=0, write_graph=True, write_images=False)
      #  self.model.fit(train_input, train_output,
       #               epochs=epochs, batch_size=batch_size,
       #               shuffle=shuffle,
       #               validation_data=(val_input, val_output),
       #               callbacks=[tensorboard])
	self.model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=200 // batch_size)
        return
    
    def encoder_predict(self, test_input):
        return self.encoder_model.predict(test_input)
    
    def predict(self, test_input):
        return self.model.predict(test_input)
    
    def save(self, model_file, encoder_model_file):
        self.model.save(model_file)
        self.encoder_model.save(encoder_model_file)
        return


train_datagen = ImageDataGenerator(
        rescale=1./255,
	
        rotation_range=40,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        '/home/ram095/sameera/retrieval/image2/data/train',  # this is the target directory
        target_size=(32, 32),  # all images will be resized to 150x150
        batch_size=1,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/home/ram095/sameera/retrieval/image2/data/validation',
        target_size=(32, 32),
        batch_size=1,
        class_mode='categorical')
"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
interested = [0, 1, 8, 9]

scrap = []
for idx, im in enumerate(x_train):
    if (y_train[idx][0] not in interested):
        scrap.append(idx)
        
x_train = np.delete(x_train, scrap, axis=0)
y_train = np.delete(y_train, scrap, axis=0)

enc = preprocessing.OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()

scrap = []
for idx, im in enumerate(x_test):
    if (y_test[idx][0] not in interested):
        scrap.append(idx)
x_test = np.delete(x_test, scrap, axis=0)
y_test = np.delete(y_test, scrap, axis=0)
y_test = enc.transform(y_test).toarray()

x_train = (x_train.astype('float32')) / 255.0
x_test = (x_test.astype('float32')) / 255.0

print x_train.shape
print x_test.shape
"""
fe = FeatureExtractor()
"""
opt = optimizers.adam(lr=0.00001, decay=1e-6)
#opt = optimizers.rmsprop()
#opt = optimizers.SGD(lr=0.1, nesterov=True, momentum=0.9)
fe.build((32, 32, 3, ), opt)

fe.train(train_generator, validation_generator,
                 epochs=1,
                 batch_size=1,
                 shuffle=True)

fe.save('/home/ram095/sameera/retrieval/image2/extractor.h5', '/home/ram095/sameera/retrieval/image2/extractor-model.h5')
print "Model saved!"
"""
img = load_img('/home/ram095/sameera/retrieval/image2/data/validation/1/test.jpg')
x = img_to_array(img.resize((32,32)))  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


fe.load('/home/ram095/sameera/retrieval/image2/extractor.h5')
prediction = fe.predict(x)
print prediction

