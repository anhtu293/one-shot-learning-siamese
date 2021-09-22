import tensorflow as tf
import numpy as np


class LRPrinter(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(LRPrinter, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        print('Learning rate at epoch {} : {:.9f}'.format(epoch, self.model.optimizer.lr.numpy()))


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, path, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.path = path

    def on_train_begin(self, logs=None):
        self.val_loss = [np.inf]

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss <= min(self.val_loss):
            print('Model improved from {} to {}'.format(min(self.val_loss), current_val_loss))
            self.model.siamese_network.save(self.path + '_{:.4f}'.format(current_val_loss), save_format='tf')
        self.val_loss.append(current_val_loss)
