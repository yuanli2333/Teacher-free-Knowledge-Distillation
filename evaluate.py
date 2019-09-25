"""Evaluates the model"""

import argparse
import logging

from torch.autograd import Variable
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, params, args):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    losses = utils.AverageMeter()
    total = 0
    correct = 0

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        # compute model output
        output_batch = model(data_batch)
        if args.regularization:
            loss = loss_fn(output_batch, labels_batch, params)
        else:
            loss = loss_fn(output_batch, labels_batch)

        losses.update(loss.data, data_batch.size(0))
        _, predicted = output_batch.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    loss_avg = losses.avg
    acc = 100.*correct/total
    logging.info("- Eval metrics, acc:{acc:.4f}, loss: {loss_avg:.4f}".format(acc=acc, loss_avg=loss_avg))
    my_metric = {'accuracy': acc, 'loss': loss_avg}
    return my_metric


"""
This function duplicates "evaluate()" but ignores "loss_fn" simply for speedup purpose.
Validation loss during KD mode would display '0' all the time.
One can bring that info back by using the fetched teacher outputs during evaluation (refer to train.py)
"""
def evaluate_kd(model, dataloader, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    total = 0
    correct = 0

    # compute metrics over the dataset
    for i, (data_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        output_batch = model(data_batch)

        # loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
        loss = 0.0  #force validation loss to zero to reduce computation time
        _, predicted = output_batch.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    acc = 100. * correct / total
    logging.info("- Eval metrics, acc:{acc:.4f}, loss: {loss:.4f}".format(acc=acc, loss=loss))
    my_metric = {'accuracy': acc, 'loss': loss}
    #my_metric['accuracy'] = acc
    return my_metric
