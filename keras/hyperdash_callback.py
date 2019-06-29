from keras.callbacks import Callback


class Hyperdash(Callback):
    def __init__(self, exp):
        super(Hyperdash, self).__init__()
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        val_loss = logs.get('val_loss')

        if val_acc is not None:
            self.exp.metric("val_acc", val_acc)
        if val_loss is not None:
            self.exp.metric("val_loss", val_loss)
