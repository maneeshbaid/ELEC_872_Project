import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
import os
import random
import tensorflow as tf

# Updated session configuration for TensorFlow 2
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Lambda, Input, Flatten, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn import metrics


def model(x_train, num_labels, tcn_filters, num_conv_filters, batch_size, num_heads, key_dim):
    """
    The proposed model with CNN layer, TCN layer, and multi-head attention layers.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - TCN_units: number of RNN units (int)
    - num_conv_filters: number of CNN filters (int)
    - batch_size: number of samples to be processed in each batch
    - num_heads: number of attention heads (int)
    - key_dim: dimension of each attention head (int)
    Returns
    - model: A Keras model
    """
    cnn_inputs = Input(shape=(x_train.shape[1], x_train.shape[2]), batch_size=batch_size, name='rnn_inputs')
    cnn_layer = Conv1D(filters=num_conv_filters, kernel_size=3, padding='same', activation='relu', name='cnn_conv1d')
    cnn_out = cnn_layer(cnn_inputs)

    sq_layer_out = cnn_out

    tcn_layer_1 = Conv1D(filters=tcn_filters, kernel_size=3, padding='causal', activation=None,
                         kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='tcn_conv1')(sq_layer_out)
    tcn_layer_1 = BatchNormalization()(tcn_layer_1)
    tcn_layer_1 = ReLU()(tcn_layer_1)
    tcn_layer_1 = Dropout(0.1)(tcn_layer_1)

    tcn_layer_2 = Conv1D(filters=tcn_filters, kernel_size=3, padding='causal', activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='tcn_conv2')(tcn_layer_1)
    tcn_layer_2 = BatchNormalization()(tcn_layer_2)
    tcn_layer_2 = ReLU()(tcn_layer_2)
    tcn_layer_2 = Dropout(0.1)(tcn_layer_2)

    tcn_layer_output = tcn_layer_2

    multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='multi_head_attention')
    attention_output = multi_head_attention(query=tcn_layer_output, value=tcn_layer_output, key=tcn_layer_output)

    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(attention_output)

    dense_layer = Dense(num_labels, activation='softmax')
    dense_layer_output = dense_layer(pooled_output)

    model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
    print(model.summary())

    return model


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCH = 10
BATCH_SIZE = 16
TCN_FILTERS = 32
CNN_FILTERS = 3
LEARNING_RATE = 1e-5
PATIENCE = 20
SEED = 0
NUM_HEADS = 4
KEY_DIM = 16
DATA_FILES = ['WISDM.npz']
MODE = 'LOTO'
BASE_DIR = '../../dataset/' + MODE + '/'
SAVE_DIR = './model_with_wisdm_cnn_tcn_multihead_attn_' + MODE + '_results'

if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(0)

    for DATA_FILE in DATA_FILES:
        data_input_file = os.path.join(BASE_DIR, DATA_FILE)
        tmp = np.load(data_input_file, allow_pickle=True)
        X = tmp['X']
        X = np.squeeze(X, axis=1)
        y_one_hot = tmp['y']
        folds = tmp['folds']

        NUM_LABELS = y_one_hot.shape[1]

        avg_acc = []
        avg_recall = []
        avg_f1 = []
        early_stopping_epoch_list = []
        y = np.argmax(y_one_hot, axis=1)

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis=3)
            X_test_ = np.expand_dims(X_test, axis=3)

            train_trailing_samples = X_train_.shape[0] % BATCH_SIZE
            test_trailing_samples = X_test_.shape[0] % BATCH_SIZE

            if train_trailing_samples != 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples != 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print(y_train.shape, y_test.shape)

            rnn_model = model(x_train=X_train_, num_labels=NUM_LABELS, tcn_filters=TCN_FILTERS, \
                              num_conv_filters=CNN_FILTERS, batch_size=BATCH_SIZE, num_heads=NUM_HEADS, key_dim=KEY_DIM)

            model_filename = SAVE_DIR + '/best_model_with_multihead_attn_' + str(DATA_FILE[0:-4]) + '_fold_' + str(
                i) + '.weights.h5'
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor='val_accuracy', save_weights_only=True,
                                         save_best_only=True), EarlyStopping(monitor='val_accuracy', patience=PATIENCE)]

            opt = optimizers.Adam(clipnorm=1.)

            rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = rnn_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1,
                                    callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1
            print('Early stopping epoch: ' + str(early_stopping_epoch))
            early_stopping_epoch_list.append(early_stopping_epoch)

            if early_stopping_epoch <= 0:
                early_stopping_epoch = -100

            print("******Evaluating TEST set*********")
            rnn_model.load_weights(model_filename)

            y_test_predict = rnn_model.predict(X_test_, batch_size=BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            all_trainable_count = int(np.sum([K.count_params(p) for p in rnn_model.trainable_weights]))

            MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            acc_fold = accuracy_score(y_test, y_test_predict)
            avg_acc.append(acc_fold)

            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            avg_recall.append(recall_fold)

            f1_fold = f1_score(y_test, y_test_predict, average='macro')
            avg_f1.append(f1_fold)

            with open(SAVE_DIR + '/results_model_with_multihead_attn_' + MODE + '.csv', 'a') as out_stream:
                out_stream.write(str(SEED) + ', ' + str(DATA_FILE[0:-4]) + ', ' + str(i) + ', ' + str(
                    early_stopping_epoch) + ', ' + str(all_trainable_count) + ', ' + str(acc_fold) + ', ' + str(
                    MAE) + ', ' + str(recall_fold) + ', ' + str(f1_fold) + '\n')

            print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
            print('______________________________________________________')
            K.clear_session()

    std_acc = np.std(avg_acc, ddof=1)
    std_recall = np.std(avg_recall, ddof=1)
    std_f1 = np.std(avg_f1, ddof=1)

    print('Accuracy: {:.2f}% ± {:.2f}'.format(np.mean(avg_acc) * 100, std_acc * 100))
    print('Recall: {:.2f}% ± {:.2f}'.format(np.mean(avg_recall) * 100, std_recall * 100))
    print('F1-Score: {:.2f}% ± {:.2f}'.format(np.mean(avg_f1) * 100, std_f1 * 100))
