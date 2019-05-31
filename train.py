from keras import optimizers
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.DataManager import batchGenerator, threadsafe_iter
from utils.Model import build_classify_model, Creat_train_Model

isTrainClassify = 0

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
}

# define monitor
def classify_acc(y_true, y_pred):
    classify_label = y_true[2:]
    classify_pred = y_pred[2:]
    return metrics.categorical_accuracy(y_true = classify_label, y_pred = classify_pred)
def filter_acc(y_true, y_pred):
    filter_label = y_true[:2]
    filter_pred = y_pred[:2]
    return metrics.categorical_accuracy(y_true = filter_label, y_pred = filter_pred)

def train():
    # Create data generator
    train_gen = batchGenerator(data_path = params['train_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = True)
    val_gen = batchGenerator(data_path = params['val_dataset_path'], input_size = params['image_size'], batch_size = params['batch_size'], random = False)
    # Get dataset information
    num_train, num_val = len(train_gen), len(val_gen)

    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(params['image_size'])
    model.compile(loss= {'loss_function': lambda y_true, y_pred: y_pred}, optimizer = optimizers.Adam(1e-3), metrics=[classify_acc, filter_acc])

    # Callbacks list
    checkpoint = ModelCheckpoint(filepath = params['filter_model_path'], monitor = 'val_filter_acc', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_filter_acc', factor=0.1, patience = 5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_filter_acc', patience = 10, verbose=1)

    # Start training
    model.fit_generator(threadsafe_iter(train_gen),
                        steps_per_epoch = max(1, num_train // params['batch_size']), 
                        validation_data = threadsafe_iter(val_gen), 
                        validation_steps = max(1, num_val // params['batch_size']),
                        epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                        callbacks = [checkpoint, reduce_lr, early_stopping])

if __name__ == "__main__":
    train()