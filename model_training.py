import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import matplotlib.pyplot as plt
retinal_vessel_data = np.load('retinal_vessel_dataset.npz')
x_train_raw = retinal_vessel_data["x_train"][...,np.newaxis]
y_train = retinal_vessel_data["y_train"][...,np.newaxis].astype(int)
x_val_raw = retinal_vessel_data["x_val"][...,np.newaxis]
y_val = retinal_vessel_data["y_val"][...,np.newaxis].astype(int)

def visualize(img1, img1caption, 
                         img2, img2caption,
                         img3, img3caption,
                         img4, img4caption,  
                         img1range = (0,255), img2range = (0,255), img3range = (0,255), img4range = (0, 255)):
  
    fig, axs = plt.subplots(1,4, figsize=(16,24))
    axs[0].imshow(img1,cmap='gray',vmax=img1range[1],vmin=img1range[0])                       
    axs[1].imshow(img2,cmap='gray',vmax=img2range[1],vmin=img2range[0])
    axs[2].imshow(img3,cmap='gray',vmax=img3range[1],vmin=img3range[0])  
    axs[3].imshow(img4,cmap='gray',vmax=img4range[1],vmin=img4range[0])                            
    axs[0].set_title(img1caption)     
    axs[1].set_title(img2caption)
    axs[2].set_title(img3caption)
    axs[3].set_title(img4caption)
    fig.show()

    
def contrast_stretch(x):
  I_max = np.max(x, axis=(1, 2))[:, :, None, None] # shape: (# of sample, 1) -> (# of sample, 1, 1, 1)
  I_min = np.min(x, axis=(1, 2))[:, :, None, None] # shape: (# of sample, 1) -> (# of sample, 1, 1, 1)
  x_enhanced = ((x - I_min)/(I_max - I_min) * 255).astype(int)
  return x_enhanced


x_train_enhanced = contrast_stretch(x_train_raw)
x_val_enhanced = contrast_stretch(x_val_raw)

def rescale_01(x):
  x_01 = x/255
  return x_01.astype(float)
  
def compile_model(model, lr):
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=lr))


def train_model(model, epochs, x_train, y_train, x_val, y_val):
    return model.fit(x=x_train, y=y_train, batch_size=16, epochs=epochs, validation_data=(x_val, y_val), validation_batch_size=16)


def predict_model(model, x_val):
    val_preds = model.predict(x_val, batch_size=16)
    return val_preds


def threshold(val_preds, thresh_value):
    val_preds_thresh = val_preds >= thresh_value
    return val_preds_thresh.astype(int)

def dice_coef(mask1, mask2):
    dice_coef_score = 2*(np.sum(mask1*mask2))/(np.sum(mask1) + np.sum(mask2))
    return dice_coef_score

def avg_dice(y_val, val_preds_thresh):
    average_dice = 0
    for i, j in zip(y_val, val_preds_thresh):
        average_dice += dice_coef(i, j)
    average_dice /= y_val.shape[0] # get the average
    return average_dice
    
x_train = rescale_01(x_train_enhanced)
x_val = rescale_01(x_val_enhanced)


# build the model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout

def build_model():
    model = Sequential()
    # abstraction path
    model.add(Conv2D(45, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal())) # TODO: specify input shape!
    model.add(Dropout(0.3))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(Dropout(0.3))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # bottleneck path
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))

    # expansion path
    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    model.add(Conv2D(50, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(Dropout(0.25))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))
    model.add(Dropout(0.25))
    model.add(Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer=keras.initializers.HeNormal()))

    # final layer
    model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer=keras.initializers.HeNormal()))

    ### END YOUR CODE HERE
    model.build((None, 64,64,1))

    return model


model = build_model()
model.summary()
    

keras.utils.set_random_seed(2211) # Reset seed, you should get the same model since this code cell

learning_rate = 0.001  # Feel free to experiment with different values!
num_epochs = 85       # Feel free to experiment with different values!
# at around 107/200 epoch, the test accuracy start to be stabled
model = build_model() # Build new model with newly initialized weights
compile_model(model, learning_rate)
train_model(model, num_epochs, x_train, y_train, x_val, y_val)

val_preds = predict_model(model, x_val)


val_preds_thresh = threshold(val_preds, 0.5)  

accuracy = "{:.4f}".format(avg_dice(y_val, val_preds_thresh))
print("accuracy:", accuracy)