from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.DataManager import batchGenerator, threadsafe_iter
from utils.Model import build_classify_model, Creat_train_Model

isTrainClassify = 1

workers = 32
max_queue_size = 1000
params = {
        "image_size" : (128, 128, 3),
        "epochs" : 30,
        "batch_size" : 32,
        

        "classify_model_path" : 'classify_model.h5',
        "filter_model_path" : 'filter_model.h5',

        "train_dataset_path" : 'data\\Training.pkl',
        "val_dataset_path" : 'data\\Validation.pkl',
        "preload_dataset" : True,
}


def train():
    # Create data generator
    train_gen = batchGenerator(data_path = params['train_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = True, image_preload = params['preload_dataset'])
    val_gen = batchGenerator(data_path = params['val_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = True, image_preload = params['preload_dataset'])
    # Get dataset information
    num_train, num_val = len(train_gen), len(val_gen)

    
    if(isTrainClassify):
        classify_model = build_classify_model(params['image_size'], ouput_feature = False)
        classify_model.summary()
        classify_model.compile(loss= 'categorical_crossentropy', optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath = params['classify_model_path'], monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

        classify_model.fit_generator(threadsafe_iter(train_gen),
                            steps_per_epoch = max(1, num_train // params['batch_size']), 
                            validation_data = threadsafe_iter(val_gen), 
                            validation_steps = max(1, num_val // params['batch_size']),
                            epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                            callbacks = [checkpoint, reduce_lr, early_stopping])

    train_gen.isClassify = False
    val_gen.isClassify = False

    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(params['image_size'], classify_model_path = params['classify_model_path'])
    model.summary()
    model.compile(loss= 'binary_crossentropy', optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

    # Callbacks list
    checkpoint = ModelCheckpoint(filepath = params['filter_model_path'], monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

    # Start training
    model.fit_generator(threadsafe_iter(train_gen),
                        steps_per_epoch = max(1, num_train // params['batch_size']), 
                        validation_data = threadsafe_iter(val_gen), 
                        validation_steps = max(1, num_val // params['batch_size']),
                        epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                        callbacks = [checkpoint, reduce_lr, early_stopping])

if __name__ == "__main__":
    train()