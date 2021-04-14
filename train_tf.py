import os
import argparse
import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == '__main__':

    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()

    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)

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
    INPUT_TENSOR_NAME = "inputs_input" # Watch out, it needs to match the name of the first layer + "_input"

    # === Generators ===

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(training_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    valid_generator = valid_datagen.flow_from_directory(validating_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)

    # === Model ===

    # ResNet structure without classification layer
    model = Sequential()
    model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet'
    ))
    
    # Output layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.layers[0].trainable = False
    
    
    # Callback
    callbacks = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator,
              validation_data=valid_generator,
              epochs=epochs,
              callbacks = [callbacks])

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

