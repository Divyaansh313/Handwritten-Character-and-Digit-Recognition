from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

def model_train(epoch,n):
    batch_size = 32
    num_classes = 10
    epochs = epoch

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if n>1:
        model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    global acc, val_acc, loss, val_loss, no_of_epochs
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    no_of_epochs = range(1,len(acc)+1)
    print(acc)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy : ', score[1]*100)
    prediction = model.predict(x_test[:1])
    print("prediction shape:", prediction.shape)
    y_pred = model.predict(x_test)
    X_test__ = x_test.reshape(x_test.shape[0], 28, 28)

    fig, axis = plt.subplots(4, 4, figsize=(12, 14))
    for i, ax in enumerate(axis.flat):
      ax.imshow(X_test__[i], cmap='binary')
      ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}");
        
        
no_epoch=10
no_layer=1
accuracy_train_model=model_train(no_epoch,no_layer)


def plotgraph(no_of_epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(no_of_epochs, acc, 'b')
    plt.plot(no_of_epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
# Accuracy curve
plotgraph(no_of_epochs,acc,val_acc)

# loss curve
plotgraph(no_of_epochs, loss, val_loss)
