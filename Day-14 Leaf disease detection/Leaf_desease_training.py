


from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import BatchNormalization # it helps to standadise the values
from keras.layers import Dropout #control the over fit

#generating the cnn

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(rate=0.2))
model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dropout(rate= 0.3))

model.add(Dense(4,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


train_datagen= ImageDataGenerator(rescale= None,
                                  shear_range= 0.2, 
                                  zoom_range=0.2,
                                  horizontal_flip=True)
#for validation set
test_datagen= ImageDataGenerator(rescale=1./ 255)


training_set= train_datagen.flow_from_directory ('data/train',target_size=(128,128),batch_size=8,class_mode='categorical')
labels = (training_set.class_indices)
print (labels)

#validation set
test_set= test_datagen.flow_from_directory ('data/val',target_size=(128,128),batch_size=8,class_mode='categorical')
labels2 = (training_set.class_indices)
print (labels2)




model.fit_generator(training_set,
                    steps_per_epoch=375,
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=125)




model_json= model.to_json()
with open ("model.json","w") as json_file:
    json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")








