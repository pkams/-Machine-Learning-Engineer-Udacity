import os
import argparse
import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == '__main__':

    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()

    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)

    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    model_dir  = args.model_dir
    training_dir   = args.train
    validating_dir = args.eval
    batch_size = args.batch_size

    # === Variables === 

    HEIGHT = 160
    WIDTH = 160
    DEPTH = 3
    NUM_CLASSES = 2

    # === Generators ===

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(training_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    valid_generator = valid_datagen.flow_from_directory(validating_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)

    # === Model ===

    model = Sequential()

    model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(HEIGHT, WIDTH, DEPTH), activation="relu", name="inputs",
                     padding="same"))
    model.add(MaxPooling2D(3,3))
    model.add(Conv2D(224,(3,3),activation='relu'))
    model.add(MaxPooling2D(3,3))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(3,3))
    
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    
    # Callback
    callbacks = [EarlyStopping(monitor = 'val_loss',patience = 10,restore_best_weights=True)]

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    model.fit(train_generator,
              validation_data=valid_generator,
              epochs=epochs,
              callbacks = callbacks)

    tf.saved_model.simple_save(
        tf.keras.backend.get_session(),
        os.path.join(model_dir, "1"),
        inputs={"inputs": model.input},
        outputs={t.name: t for t in model.outputs})

    # Save the model 
    #model.save(os.path.join(sm_model_dir, "tf_model"), save_format="tf")

    #def model_fn(model_dir):
    #    classifier = tf.keras.models.load_model(os.path.join(sm_model_dir, "tf_model"))
    #    return classifier

