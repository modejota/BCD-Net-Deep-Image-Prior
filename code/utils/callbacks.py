import pandas as pd

class Callback():
    def __init__(self):
        pass

    def on_train_step_begin(self, step, logs=None):
        """Called at the beginning of a training step."""
        pass
    
    def on_train_step_end(self, step, logs=None):
        """Called at the end of a training step."""
        pass
    
    def on_val_step_begin(self, step, logs=None):
        """Called at the beginning of a training step."""
        pass
    
    def on_val_step_end(self, step, logs=None):
        """Called at the end of a training step."""
        pass

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of a training epoch."""
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of a training epoch."""
        pass

    def on_train_begin(self, logs=None):
        "Called at the beginning of training."
        pass
    
    def on_train_end(self, logs=None):
        "Called at the end of training."
        pass

class CallbacksList(Callback):
    def __init__(self, callbacks=[]):
        super().__init__()
        self.callbacks = callbacks
    
    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_step_begin(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_train_step_begin(step, logs)
    
    def on_train_step_end(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_train_step_end(step, logs)
    
    def on_val_step_begin(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_val_step_begin(step, logs)
    
    def on_val_step_end(self, step, logs=None):
        for callback in self.callbacks:
            callback.on_val_step_end(step, logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

class EarlyStopping(Callback):
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, model, score_name, mode="min", patience=10, delta=0.0, path="", verbose=True):
        """
        Args:
            model: model to monitor
            mode (str): one of {min, max}
            score_name (str): name of the score to monitor
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.model = model
        self.score_name = score_name
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        #self.early_stop = False
        
    
    def on_epoch_end(self, epoch, logs=None):
        if self.mode == "max":
            score = -logs[self.score_name]
        else:
            score = logs[self.score_name]

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(score)
        elif score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}.')
            if self.counter >= self.patience:
                self.model.stop_training = True
                print(f"Reached max patience. Stopping training at epoch {epoch}.")            

    def save_checkpoint(self):
        '''
        Saves model
        '''
        name = self.path + "best.pt"
        self.model.save(name)
        if self.verbose:
            print(f'Best score improved. Model saved to {self.path}.')
    
class ModelCheckpoint(Callback):
    """
    Saves model after each epoch.
    """
    def __init__(self, model, path, save_freq=10, verbose=True):
        """
        Args:
            model: model to monitor
            path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each model saved.
        """
        self.model = model
        self.path = path
        self.verbose = verbose
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            name = self.path + f"epoch{epoch}.pt"
            self.model.save(name)
            if self.verbose:
                print(f"Model saved to {self.path}.")

class History(Callback):
    def __init__(self, path, verbose=True):
        self.history = {"epoch" : []}
        self.path = path
        self.verbose = verbose
    
    def get_history(self):
        return self.history
    
    def on_epoch_end(self, epoch, logs=None):
        self.history["epoch"].append(epoch)
        if logs is not None:
            for k, v in logs.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)
    
    def on_train_end(self, logs=None):
        if self.verbose:
            print(f"Saving history to {self.path}.")
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.path, index=False)