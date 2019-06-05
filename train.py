import keras.backend as K
from keras import optimizers
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.DataManager import batchGenerator, threadsafe_iter
from utils.Model import build_classify_model, Creat_train_Model, loss_function

workers = 32
max_queue_size = 1000
params = {
        "image_size" : (128, 128, 3),
        "epochs" : 30,
        "batch_size" : 16,
        
        "classify_model_path" : 'classify_model.h5',
        "filter_model_path" : 'filter_model.h5',

        "train_dataset_path" : 'data\\Training.pkl',
        "val_dataset_path" : 'data\\Validation.pkl',
}

def train():
    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(params['image_size'])
    model.compile(loss = loss_function, optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

    # Create data generator
    train_gen = batchGenerator(data_path = params['train_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = True)
    val_gen = batchGenerator(data_path = params['val_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = False)
    # Get dataset information
    num_train, num_val = len(train_gen), len(val_gen)

    # Callbacks list
    checkpoint = ModelCheckpoint(filepath = params['filter_model_path'], monitor = 'val_filter_output_acc', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1)

    # Start training
    model.fit_generator(threadsafe_iter(train_gen),
                        steps_per_epoch = max(1, num_train // params['batch_size']), 
                        validation_data = threadsafe_iter(val_gen), 
                        validation_steps = max(1, num_val // params['batch_size']),
                        epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                        callbacks = [checkpoint, reduce_lr, early_stopping])

if __name__ == "__main__":
    train()