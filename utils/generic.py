import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


class FileOutputDuplicator(object):
    """
    Class to duplicate the output to a file and to console.
    """

    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
        self.duplicate.flush()


def predict_test_set(model, test_set, device, num_workers):
    """
    Predict on test-set samples.
    """
    print("Making prediction on test-set samples")

    if isinstance(model, tuple) or isinstance(model, list):
        for m in model:
            m.eval()
        
        # model.eval()
        test_set = test_set.eval()
        dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers)
        preds = []
        actuals = []
        with torch.no_grad():
            for idx, (x, y, _) in enumerate((dataloader)):
                pred = []
                for m in model:
                    pred.append(m(x.to(device)).detach().cpu())
                pred = torch.stack(pred)
                pred = torch.mean(pred, dim=0)
                
                preds.append(pred)
                actuals.append(y)

        preds = torch.cat(preds, dim=0)
        actuals = torch.cat(actuals, dim=0)#.numpy()
        preds = torch.argmax(preds, dim=1)#.numpy()

        return preds.numpy(), preds, actuals

    else:
        model.eval()
        test_set = test_set.eval()
        dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=num_workers)
        preds = []
        actuals = []
        with torch.no_grad():
            for idx, (x, y, _) in enumerate((dataloader)):
                pred = model(x.to(device)).detach().cpu()
                preds.append(pred)
                actuals.append(y)

        preds = torch.cat(preds, dim=0)
        actuals = torch.cat(actuals, dim=0)#.numpy()
        preds = torch.argmax(preds, dim=1)#.numpy()

        return preds.numpy(), preds, actuals


def evaluate(test_set, model, device, exp_idx, preds_file, num_workers):
    """
    Call prediction function on test-set samples and append to file.
    """

    predictions, preds, gts = predict_test_set(model, test_set, device, num_workers)

    predictions_dict = {}

    if os.path.exists(preds_file):
        with open(preds_file, "rb") as f:
            predictions_dict = pickle.load(f)

    predictions_dict[str(exp_idx)] = predictions

    with open(preds_file, "wb") as f:
        pickle.dump(predictions_dict, f)
    
    return preds, gts