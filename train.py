from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.DataManager import batchGenerator, threadsafe_iter
from utils.Model import build_classify_model, Creat_train_Model

isTrainClassify = 1

BATCH_SIZE = 32
input_shape = (128, 128, 3)
classify_model_path = 'classify_model.h5'
filter_model_path = 'filter_model.h5'

workers = 16
max_queue_size = 500

def train():
    train_gen = batchGenerator(input_size = input_shape, batch_size = BATCH_SIZE, random = True)
    num_train, num_val = train_gen.train_set_len(), train_gen.val_set_len()
    print('Training data num: %d' % num_train)
    # print('Validation data num: %d' % num_val)
    

    if(isTrainClassify):
        train_gen.isClassify = True
        classify_model = build_classify_model(input_shape)
        classify_model.summary()
        classify_model.compile(loss= 'categorical_crossentropy', optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath = classify_model_path, monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

        gen = threadsafe_iter(train_gen, train_gen.train_flow())
        val_gen = threadsafe_iter(train_gen, train_gen.val_flow())
        classify_model.fit_generator(gen,
                            steps_per_epoch = max(1, num_train // BATCH_SIZE), 
                            validation_data = val_gen, 
                            validation_steps = max(1, num_val // BATCH_SIZE),
                            epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                            callbacks = [checkpoint, reduce_lr, early_stopping])

    train_gen.isClassify = False
    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape, classify_model_path = classify_model_path)
    model.summary()
    model.compile(loss= 'binary_crossentropy', optimizer = optimizers.Adam(1e-3), metrics=['accuracy'])

    # Callbacks list
    checkpoint = ModelCheckpoint(filepath = filter_model_path, monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

    # Start training
    gen = threadsafe_iter(train_gen, train_gen.train_flow())
    val_gen = threadsafe_iter(train_gen, train_gen.val_flow())
    model.fit_generator(gen,
                        steps_per_epoch = max(1, num_train // BATCH_SIZE), 
                        validation_data = val_gen, 
                        validation_steps = max(1, num_val // BATCH_SIZE),
                        epochs = 1000, verbose=1, workers = workers, max_queue_size = max_queue_size,
                        callbacks = [checkpoint, reduce_lr, early_stopping])

if __name__ == "__main__":
    train()