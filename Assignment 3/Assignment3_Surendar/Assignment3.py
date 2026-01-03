import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir='archive'

img_aug=ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.2
)
val_aug=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
####------AUGMENTED DATA LOADING--------#####
training_data=img_aug.flow_from_directory(
    data_dir, target_size=(128,128),
    batch_size=32,class_mode='binary',subset='training'
)

val_data=val_aug.flow_from_directory(
    data_dir, target_size=(128,128),
    batch_size=32,class_mode='binary',subset='validation'
)

###------CNN MODEL BUILDING--------####

model=keras.Sequential([
    keras.layers.Input(shape=(128,128,3)),

    ##FIRST LAYER OF CONV
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    ##SECOND LAYER OF CONV
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    #THIRD LAYER OF CONV
    layers.Conv2D(128,(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

##TRAINING MODEL
history=model.fit(
    training_data,
    validation_data=val_data,
    epochs=15
)

###PLOTTING GRAPH FOR ACCURACY AND LOSS AT VARIOUD EPOCHS
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']


###-------------AS WE GOT A PLATEUED VAL_ACC ,IT MAY BE DUE TO CLASS IMBALANCE------------#####


##VERIFYING IF THERE IS AN IMBALANCE:
def check_imbalance(generator, name):
    labels=generator.classes
    unique,counts=np.unique(labels,return_counts=True)

    print(f'###---{name} SET----####')
    total_img=sum(counts)

    print(f"TOTAL IMAGES: {total_img}")
    for i,count in zip(unique,counts):
        class_name = list(generator.class_indices.keys())[i]
        perc=(count/total_img)*100
        print(f"{class_name}: {count} images ({perc:.2f}%)")
    print('\n')

check_imbalance(training_data,'TRAINING')
check_imbalance(val_data,'VALIDATION')

##HERE WE WERE ABLE TO ANALYSE THAT HEAVY IMBALANCE WAS THERE IN BOTH TRAINING AND TESTING
## SO INORDER TO SOLVE THIS PROBLEM,LETS PENALIZE THE MODEL HEAVILY EVERYTIME IT PREDICTS THE 0s WRONG

from sklearn.utils import class_weight

model_new = keras.Sequential([
    keras.layers.Input(shape=(128, 128, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),  ##ADDING STABLIZER TO PREVENT HIGH FLUCTUATIONS
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),


    layers.Flatten(),
    layers.Dropout(0.3),  # Lowered from 0.5 to 0.3
    layers.Dense(1, activation='sigmoid')
])

labels=training_data.classes
weights=class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

weights_dict = dict(enumerate(weights))
print("TRAINING THE MODEL AGAIN WITH PENALIZED WEIGHTS")

optimizer_new=tf.keras.optimizers.Adam(learning_rate=0.0001)
model_new.compile(
    optimizer=optimizer_new,loss='binary_crossentropy',
    metrics=['accuracy']
)
history_new=model_new.fit(training_data,
                      validation_data=val_data,
                      epochs=25,
                      class_weight=weights_dict)

acc_new=history_new.history['accuracy']
val_acc_new=history_new.history['val_accuracy']
loss_new=history_new.history['loss']
val_loss_new=history_new.history['val_loss']

###---------PLOTTING GRAPHS TO ANALYSE VISUALLY-------#####
fig,axs=plt.subplots(2,2,figsize=(15,10))
axs[0,0].plot(
    np.arange(1,16),
    acc,
    label='Training Accuracy',
)
axs[0,0].plot(
    np.arange(1,16),
    val_acc,
    label='Validation Accuracy'
)
axs[0,0].set_title('Accuracy(WITHOUT BALANCED WEIGHTS)',fontweight='bold',color='red')
axs[0,0].legend(loc='lower right')

axs[0,1].plot(
    np.arange(1,16),
    loss,
    label='Training Loss'
)
axs[0,1].plot(
    np.arange(1,16),
    val_loss,
    label='Validation Loss'
)
axs[0,1].set_title('Loss(WITHOUT BALANCED WEIGHTS',fontweight='bold',color='red')
axs[0,1].legend(loc='lower right')

axs[1,0].plot(
    np.arange(1,26),
    acc_new,
    label='Training Accuracy'
)
axs[1,0].plot(
    np.arange(1,26),
    val_acc_new,
    label='Validation Accuracy'
)
axs[1,0].set_title('Accuracy(WITH BALANCED WEIGHTS)',fontweight='bold',color='darkblue')
axs[1,0].legend(loc='lower right')

axs[1,1].plot(
    np.arange(1,26),
    loss_new,
    label='Training Loss'
)
axs[1,1].plot(
    np.arange(1,26),
    val_loss_new,
    label='Validation Loss'
)
axs[1,1].set_title('Loss(WITH BALANCED WEIGHTS)',fontweight='bold',color='darkblue')
axs[1,1].legend(loc='lower right')
plt.show()
print('##-----OVER AND OUT------##')