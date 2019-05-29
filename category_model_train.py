from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.Category_Model import CreatModel
from utils.Category_DataManager import batchGenerator

BATCH_SIZE = 32
input_shape = (64, 64, 3)

def train():
    train_gen = batchGenerator(input_size = input_shape, batch_size = BATCH_SIZE, aug = True)

    num_train, num_val = train_gen.train_set_len(), train_gen.val_set_len() # Print the dataset size
    print('Training data num: %d' % num_train)
    # print('Validation data num: %d' % num_val)

    # Create the model
    model = CreatModel(input_shape = input_shape, output_shape = 8)
    model.summary()
    model.compile(loss= 'categorical_crossentropy', optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

    # Callbacks list
    save_model_path = 'category_model.h5'

    # model.load_weights(save_model_path)

    checkpoint = ModelCheckpoint(filepath = save_model_path, monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

    # Start training
    model.fit_generator(train_gen.train_flow(),
                        steps_per_epoch = max(1, num_train // BATCH_SIZE), 
                        validation_data = train_gen.val_flow(), 
                        validation_steps = max(1, num_val // BATCH_SIZE),
                        epochs = 1000, verbose=1,
                        callbacks = [checkpoint, reduce_lr, early_stopping])

if __name__ == "__main__":
    train()