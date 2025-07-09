import os
import sys
import numpy as np
import logging

import soundfile as sf
import tqdm

import torch
import torch.nn.functional as F

from math import ceil

from read_emg import SizeAwareSampler
from dataset import EMGEMADataset
from model import EMG2EMA
from align import align_from_distances
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'weight of auxiliary phoneme prediction loss')
flags.DEFINE_float('pitch_loss_weight', 0.1, 'weight of pitch prediction loss')
flags.DEFINE_float('loudness_loss_weight', 0.8, 'weight of loudness prediction loss')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output', 'output directory')
flags.DEFINE_boolean('pitch_norm', False, 'normalize pitch')
flags.DEFINE_integer('pitch_minus', 130, 'subtract from pitch')
flags.DEFINE_integer('pitch_div', 25, 'divide from pitch')
flags.DEFINE_boolean('loudness_norm', False, 'normalize loudness')
flags.DEFINE_float('ema_resample', 86.16, 'target resample rate for EMA')
flags.DEFINE_float('emg_resample', 689.06, 'target resample rate for EMG')
flags.DEFINE_boolean('decollate', True, 'combine fixed length then decollate')


def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_raw)
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))
    seq_len = 200
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*ceil(FLAGS.emg_resample/FLAGS.ema_resample))
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred_ema, pred_pitch, pred_loudness, phoneme_pred = model(X, X_raw, sess)

            loss, phon_acc = dtw_loss(pred_ema, pred_pitch, pred_loudness, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())

            accuracies.append(phon_acc)

    model.train()

    return np.mean(losses), np.mean(accuracies), phoneme_confusion #TODO size-weight average


def get_aligned_prediction(model, datapoint, device, audio_normalizer):
    model.eval()
    with torch.no_grad():
        silent = datapoint['silent']
        sess = datapoint['session_ids'].to(device).unsqueeze(0)
        X = datapoint['emg'].to(device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(device).unsqueeze(0)
        y = datapoint['parallel_voiced_audio_features' if silent else 'audio_features'].to(device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess) # (1, seq, dim)

        if silent:
            costs = torch.cdist(pred, y).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            pred_aligned = pred.squeeze(0)[alignment]
        else:
            pred_aligned = pred.squeeze(0)

        pred_aligned = audio_normalizer.inverse(pred_aligned.cpu())

    model.train()
    return pred_aligned

def dtw_loss(pred_ema, pred_pitch, pred_loudness, phoneme_predictions, example, phoneme_eval=False, phoneme_confusion=None):
    device = pred_ema.device

    pred_ema = decollate_tensor(pred_ema, example['ema_lengths'])
    pred_loudness = decollate_tensor(pred_loudness, example['ema_lengths'])
    pred_pitch = decollate_tensor(pred_pitch, example['ema_lengths'])

    phoneme_predictions = decollate_tensor(phoneme_predictions, example['phoneme_lengths'])

    ema = [t.to(device, non_blocking=True) for t in example['ema']]
    pitch = [t.to(device, non_blocking=True) for t in example['pitch']]
    loudness = [t.to(device, non_blocking=True) for t in example['loudness']]

    phoneme_targets = example['phonemes']

    losses = []
    correct_phones = 0
    total_length = 0

    for pr_ema, pr_pitch, pr_loudness, y_ema, y_pitch, y_loudness, pred_phone, y_phone, silent in zip(pred_ema, pred_pitch, pred_loudness, ema, pitch, loudness, phoneme_predictions, phoneme_targets, example['silent']):
        y_phone = y_phone.to(device)

        if silent: # silent mode not supported curently
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs = dists.squeeze(0)

            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:,y_phone]

            costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs

            alignment = align_from_distances(costs.T.cpu().detach().numpy())

            loss = costs[alignment,range(len(alignment))].sum()

            if phoneme_eval:
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()

                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        else:
            assert y_ema.size(0) == pr_ema.size(0)

            ema_dists = F.pairwise_distance(y_ema, pr_ema)
            pitch_dists = F.pairwise_distance(y_pitch, pr_pitch)
            loudness_dists = F.pairwise_distance(y_loudness, pr_loudness)

            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1

            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = ema_dists.sum() \
                    + FLAGS.pitch_loss_weight * pitch_dists.sum() \
                    + FLAGS.loudness_loss_weight * loudness_dists.sum() \
                    + FLAGS.phoneme_loss_weight * phoneme_loss

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()

                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y_ema.size(0)

    return sum(losses)/total_length, correct_phones/total_length

def train_model(trainset, devset, device):
    n_epochs = FLAGS.epochs

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)
    dataloader = torch.utils.data.DataLoader(training_subset, pin_memory=(device=='cuda'), collate_fn=devset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(training_subset, 256000))

    n_phones = len(phoneme_inventory)
    model = EMG2EMA(devset.num_features, 12, n_phones).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    seq_len = 200

    batch_idx = 0
    for epoch_idx in range(n_epochs):
        losses = []
        val, phoneme_acc, _ = test(model, devset, device)

        for batch in tqdm.tqdm(dataloader, 'Train step', disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)


            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*ceil(FLAGS.emg_resample/FLAGS.ema_resample))
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)


            pred_ema, pred_pitch, pred_loudness, phoneme_pred = model(X, X_raw, sess)

            loss, _ = dtw_loss(pred_ema, pred_pitch, pred_loudness, phoneme_pred, batch)
            losses.append(loss.item())

            loss.backward() 
            optim.step()

            batch_idx += 1
        train_loss = np.mean(losses)

        lr_sched.step(val)
        logging.info(f'finished epoch {epoch_idx} - validation loss: {val:.4f} training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc*100:.2f}')
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,f'model_{epoch_idx+1}.pt'))

        evaluate(devset, FLAGS.output_directory)

    return model

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(sys.argv)

    with open(os.path.join(FLAGS.output_directory, 'flag_config.txt'), 'w') as f:
        for k,v in FLAGS.flag_values_dict().items():
            if isinstance(v, list):  # Convert list to a comma-separated string
                v = ','.join(map(str, v))
            if isinstance(v, bool):
                v = str(v).lower()
            if v is None:
                continue
            line = f'--{k}={v}\n'
            print(line, end='')
            f.write(f'--{k}={v}\n')
        print(f"FLAGS saved to {os.path.join(FLAGS.output_directory, 'flag_config.txt')}")

    trainset = EMGEMADataset(dev=False,test=False)
    devset = EMGEMADataset(dev=True, test=False)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(trainset, devset, device)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
