import torch
import numpy as np
from functools import lru_cache
import json
import os
import re
import random
import pickle
from copy import copy
import sys

from read_emg import EMGDirectory, apply_to_all, notch_harmonics, remove_drift, subsample
from data_utils import load_audio, get_emg_features, phoneme_inventory, read_phonemes, TextTransform
from absl import flags

FLAGS = flags.FLAGS


def load_arti(filename, start=None, end=None, max_frames=None):

    with open(filename, 'rb') as f:
        arti_raw = pickle.load(f)
    
    ema = arti_raw['ema']
    loudness = arti_raw['loudness']
    pitch = arti_raw['pitch']

    pitch_mean, pitch_std, loudness_mean, loudness_std = None, None, None, None

    if FLAGS.pitch_norm:
        pitch_mean, pitch_std = arti_raw['pitch_stats']
        if True: # approximate pitch mean and std for easier reconstruction
            pitch_mean = FLAGS.pitch_minus
            pitch_std = FLAGS.pitch_div
        if pitch_std:
            pitch = (pitch - pitch_mean) / pitch_std
            # print('pitch', pitch_mean, pitch_std)
        else:
            pitch = np.zeros(pitch.shape)

    if FLAGS.loudness_norm:
        loudness_mean = np.mean(loudness)
        loudness_std = np.std(loudness)
        loudness = (loudness - loudness_mean) / loudness_std

    return ema, (loudness, loudness_mean, loudness_std), (pitch, pitch_mean, pitch_std)

def load_emg_ema(base_dir, index, limit_length=False, debug=False, text_align_directory=None):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]

    emg_orig = apply_to_all(subsample, x, FLAGS.emg_resample, 1000)
    x = apply_to_all(subsample, x, 516.79, 1000)
    emg = x


    if FLAGS.remove_channels:
        channels_to_keep = [c for c in range(8) if str(c) not in FLAGS.remove_channels]
        emg_orig = emg_orig[:,channels_to_keep]

        for c in FLAGS.remove_channels:
            emg[:,int(c)] = 0



    emg_features = get_emg_features(emg)

    ema, (loudness, loudness_mean, loudness_std), (pitch, pitch_mean, pitch_std) = load_arti(os.path.join(base_dir, f'{index}_audio_clean_ema.pkl'))

    if FLAGS.ema_resample != 50:
        ema = apply_to_all(subsample, ema, FLAGS.ema_resample, 50) # to match mel frequencey domain
        loudness = apply_to_all(subsample, loudness, FLAGS.ema_resample, 50)
        pitch = apply_to_all(subsample, pitch, FLAGS.ema_resample, 50)

    mfccs = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))


    shortest_len = min(mfccs.shape[0], emg_features.shape[0], ema.shape[0], pitch.shape[0], loudness.shape[0], )

    mfccs = mfccs[:shortest_len, :]
    emg_features = emg_features[:shortest_len, :]
    ema = ema[:shortest_len, :]
    pitch = pitch[:shortest_len, :]
    loudness = loudness[:shortest_len, :]

    if int(FLAGS.ema_resample) != 50:
        emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    else:
        pass
        reduce_factor = 1
        for strd in FLAGS.stride_sizes:
            reduce_factor *= int(strd)
        emg_orig = emg_orig[reduce_factor:reduce_factor+reduce_factor*shortest_len, :]

        if ema.shape[0] > emg_orig.shape[0]/reduce_factor:
            ema = ema[:int(emg_orig.shape[0]/reduce_factor)]
            pitch = pitch[:int(emg_orig.shape[0]/reduce_factor)]
            loudness = loudness[:int(emg_orig.shape[0]/reduce_factor)]

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, ema.shape[0])
    else:
        phonemes = np.zeros(shortest_len, dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32), [ema, loudness, pitch]


class EMGEMADataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = FLAGS.text_align_directory

        if no_testset:
            devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in FLAGS.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(FLAGS.silent_data_directories) > 0
            for vd in FLAGS.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(FLAGS.normalizers_file,'rb'))

        sample_mfccs, sample_emg, _, _, _, _, _ = load_emg_ema(self.example_indices[0][0].directory, self.example_indices[0][1])
        if sample_mfccs is not None:
            self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.text_transform = TextTransform()

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, emg, text, book_location, phonemes, raw_emg, (ema, loudness, pitch) = load_emg_ema(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)
        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)

        if not self.no_normalizers:
            if mfccs is not None:
                mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {'audio_features':torch.from_numpy(mfccs).pin_memory(), 'emg':torch.from_numpy(emg).pin_memory(), 'text':text, 'text_int': torch.from_numpy(text_int).pin_memory(), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids).pin_memory(), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg).pin_memory(),
                  'ema':torch.from_numpy(ema).pin_memory(), 'loudness':torch.from_numpy(loudness).pin_memory(), 'pitch':torch.from_numpy(pitch).pin_memory(),
                  'filename':os.path.join(directory_info.directory, f'{idx}_audio_clean_ema.pkl')}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg, _, _, phonemes, _, _ = load_emg_ema(voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs).pin_memory()
            result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg).pin_memory()

            audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'

        result['phonemes'] = torch.from_numpy(phonemes).pin_memory() # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file

        return result

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        ema = []
        ema_lengths = []
        pitch = []
        loudness = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
                ema.append(ex['ema'])
                ema_lengths.append(ex['ema'].shape[0])
                pitch.append(ex['pitch'])
                loudness.append(ex['loudness'])

        phonemes = [ex['phonemes'] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        raw_emg = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        text_ints = [ex['text_int'] for ex in batch]
        text_lengths = [ex['text_int'].shape[0] for ex in batch]
        filenames = [ex['filename'] for ex in batch]
        phoneme_lengths = [ex['phonemes'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'ema':ema,
                  'ema_lengths':ema_lengths,
                  'pitch':pitch,
                  'loudness':loudness,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'silent':silent,
                  'text_int':text_ints,
                  'text_int_lengths':text_lengths,
                  'filename':filenames,
                  'phoneme_lengths':phoneme_lengths}
        return result


if __name__=='__main__':
    FLAGS(sys.argv)

    trainset = EMGEMADataset(dev=False,test=False)
    devset = EMGEMADataset(dev=True, test=False)
    testset = EMGEMADataset(dev=False, test=True)

    print(trainset.__len__(), devset.__len__(), testset.__len__())