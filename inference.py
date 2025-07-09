import os
import sys
import tqdm
import pickle as pkl
import torch 
import numpy as np
from read_emg import apply_to_all, subsample
from model import EMG2EMA
from dataset import EMGEMADataset
from absl import flags
from data_utils import phoneme_inventory
from train import dtw_loss

FLAGS = flags.FLAGS
flags.DEFINE_integer('checkpoint_idx', 1, 'checkpoint index')
flags.DEFINE_string('test_type', 'test', 'dev or test')
non_zero_only = False

import torch

def calculate_pearson_correlation(x, y): 
    """
    Calculate the Pearson correlation coefficient for two 1D tensors. (ChatGPT generated)

    Args:
        x (torch.Tensor): First time series (1D tensor).
        y (torch.Tensor): Second time series (1D tensor).

    Returns:
        torch.Tensor: Pearson correlation coefficient.
    """
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape. {x.shape, y.shape}")

    # Compute mean of x and y
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Center the data by subtracting the mean
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Calculate the numerator (covariance)
    numerator = torch.sum(x_centered * y_centered)

    # Calculate the denominator (product of standard deviations)
    denominator = torch.sqrt(torch.sum(x_centered**2)) * torch.sqrt(torch.sum(y_centered**2))

    if denominator == 0:
        raise ValueError("Standard deviation of one or both series is zero, correlation is undefined.")

    # Pearson correlation
    correlation = numerator / denominator

    correlation = correlation.detach().cpu().numpy()

    return correlation


def get_correlation(preds, targets):

    pred_ema, pred_pitch, pred_loueness = preds
    ema, pitch, loudness = targets

    ema = ema.cuda()
    pitch = pitch.cuda()
    loudness = loudness.cuda()
    ema_corrs = []

    for i in range(12):
        ema_corrs.append(calculate_pearson_correlation(ema[:,i], pred_ema[0,:,i]))
    
    if non_zero_only:
        non_zero_pitch_indices = pitch[:,0] != 0
        non_zero_loudness_indices = loudness[:,0] != 0

        pitch_corr = calculate_pearson_correlation(pitch[non_zero_pitch_indices,0], pred_pitch[0,non_zero_pitch_indices,0])
        loudness_corr = calculate_pearson_correlation(loudness[non_zero_loudness_indices,0], pred_loueness[0,non_zero_loudness_indices,0])

    else:
        pitch_corr = calculate_pearson_correlation(pitch[:,0], pred_pitch[0,:,0])
        loudness_corr = calculate_pearson_correlation(loudness[:,0], pred_loueness[0,:,0])



    return ema_corrs, pitch_corr, loudness_corr

def save_articodec(ema, loudness, pitch, filename):


    ema = ema.detach().cpu().numpy().squeeze()
    loudness = loudness.detach().cpu().numpy().squeeze()
    pitch = pitch.detach().cpu().numpy().squeeze()
    filename = filename[0]

    if FLAGS.ema_resample != 50:
        ema = apply_to_all(subsample, ema, 50, FLAGS.ema_resample)
        loudness = subsample(loudness, 50, FLAGS.ema_resample)
        pitch = subsample(pitch, 50, FLAGS.ema_resample)

    with open(filename, 'rb') as f:
        original_codec = pkl.load(f)

    pred_codec = original_codec
    pred_codec['ema'] = ema
    pred_codec['loudness'] = loudness
    pred_codec['pitch'] = pitch * FLAGS.pitch_div + FLAGS.pitch_minus

    test_or_dev = 'dev' if test_type_is_dev else 'test'

    out_dir = os.path.join(FLAGS.output_directory, f'pred_codec_{FLAGS.checkpoint_idx}', test_or_dev)
    os.makedirs(out_dir, exist_ok=True)

    save_dir = os.path.join(out_dir, filename.replace('./','').replace('/','_=_').replace('.pkl','_pred.pkl'))
    print(save_dir)
    with open(save_dir, 'wb') as f:
        pkl.dump(pred_codec, f)

    return 

def interence_ema(dataset, device):

    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_raw, num_workers=0,batch_size=1)
    n_phones = len(phoneme_inventory)
    model = EMG2EMA(dataset.num_features, 12, n_phones).to(device)
    model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory,f'model_{FLAGS.checkpoint_idx}.pt')))
    model.eval()
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))

    ema_corrs = []
    pitch_corrs = []
    loudness_corrs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            sess = batch['session_ids'][0].to(device=device).unsqueeze(0)
            X = batch['emg'][0].to(dtype=torch.float32, device=device).unsqueeze(0)
            X_raw = batch['raw_emg'][0].to(dtype=torch.float32, device=device).unsqueeze(0)

            pred_ema, pred_pitch, pred_loudness, phoneme_pred = model(X, X_raw, sess)
            loss, phon_acc = dtw_loss(pred_ema, pred_pitch, pred_loudness, phoneme_pred, batch, True, phoneme_confusion)
            ema_corr, pitch_corr, loudness_corr = get_correlation([pred_ema, pred_pitch, pred_loudness], [batch['ema'][0], batch['pitch'][0], batch['loudness'][0]])
            losses.append(loss.item())
            accuracies.append(phon_acc)

            ema_corrs.append(ema_corr)
            pitch_corrs.append(pitch_corr)
            loudness_corrs.append(loudness_corr)

            save_articodec(pred_ema, pred_loudness, pred_pitch, batch['filename'])

    header = ['TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY', 'pitch', 'loudness']

    ema_corr_mean, ema_corr_std = np.mean(ema_corrs, axis=0), np.std(ema_corrs, axis=0)
    print(ema_corr_mean, ema_corr_std)

    pitch_corr_mean, pitch_corr_std = np.mean(pitch_corrs, axis=0), np.std(pitch_corrs, axis=0)
    print(pitch_corr_mean, pitch_corr_std)

    loudness_corr_mean, loudness_corr_std = np.mean(loudness_corrs, axis=0), np.std(loudness_corrs, axis=0)
    print(loudness_corr_mean, loudness_corr_std)

    test_or_dev = 'dev' if test_type_is_dev else 'test'
    stat_out_dir = os.path.join(FLAGS.output_directory, f'pred_codec_{FLAGS.checkpoint_idx}', f'0_{test_or_dev}_stats_corr.txt')

    to_write = []

    for i, he in enumerate(header):
        if he == 'pitch':
            to_write.append('\t'.join([he, str(pitch_corr_mean), str(pitch_corr_std)]))
        elif he =='loudness':
            to_write.append('\t'.join([he, str(loudness_corr_mean), str(loudness_corr_std)]))
        else:    
            to_write.append('\t'.join([he, str(ema_corr_mean[i]), str(ema_corr_std[i])]))
    
    to_write.append(f'N={len(dataset)}')

    with open(stat_out_dir,'w') as f:
        f.writelines('\n'.join(to_write))

def main():

    global test_type_is_dev
    test_type_is_dev = FLAGS.test_type == 'dev'
    dataset = EMGEMADataset(dev=test_type_is_dev, test=not test_type_is_dev)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    interence_ema(dataset=dataset, device=device)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()