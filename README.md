# Articulatory Feature Prediction from Surface EMG during Speech Production (Interspeech 2025)
### Jihwan Lee<sup>1</sup>, Kevin Huang<sup>1</sup>, Kleanthis Avramidis<sup>1</sup>, Simon Pistrosch<sup>2</sup>, Monica Gonzalez-Machorro<sup>2</sup>, Yoonjeong Lee<sup>1</sup>, Björn Schuller<sup>2</sup>, Louis Goldstein<sup>3</sup>, Shrikanth Narayanan<sup>1</sup>

#### <sup>1</sup>Signal Analysis and Interpretation Laboratory, University of Southern California, USA <br> <sup>2</sup>CHI – Chair of Health Informatics, Technical University of Munich, Germany<br> <sup>3</sup>Department of Linguistics, University of Southern California, USA

### Code implementation of [paper](https://arxiv.org/abs/2505.13814 "paper link") (Interspeech 2025)

### Abstract
We present a model for predicting articulatory features from surface electromyography (EMG) signals during speech production. The proposed model integrates convolutional layers and a Transformer block, followed by separate predictors for articulatory features. Our approach achieves a high prediction correlation of approximately 0.9 for most articulatory features. Furthermore, we demonstrate that these predicted articulatory features can be decoded into intelligible speech waveforms. To our knowledge, this is the first method to decode speech waveforms from surface EMG via articulatory features, offering a novel approach to EMG-based speech synthesis. Additionally, we analyze the relationship between EMG electrode placement and articulatory feature predictability, providing knowledge-driven insights for optimizing EMG electrode configurations. The source code and decoded speech samples are publicly available.

<!-- ![overall_architecture](figures/emg_archi.png) -->

## Sample Page

Speech samples are available [here](https://lee-jhwn.github.io/IS25-emg-ema/ "sample page").

## Data
We use the identical dataset as the following works: [Digital Voicing of Silent Speech (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.445.pdf), [An Improved Model for Voicing Silent Speech (ACL 2021)](https://aclanthology.org/2021.acl-short.23.pdf), and the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf). To download the dataset, please refer to their [official repository](https://github.com/dgaddy/silent_speech). This repository follows the same dataset structure.

## Environment Setup
Refer to `environment.yml` for the environment setting.  If you’re using Anaconda, you can create the environment with:
```sh
conda env create -f environment.yml
conda activate emg_ema
```

## Pre-processing
Articulatory features are estimated from audio using the [SPARC (Speech Articulatory Coding) model](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding). Refer to [SPARC repository](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) for more details on acoustics-to-articulatory inversion. Below is an example code snippet demonstrating how to estimate articulatory features from audio using SPARC.
```python
from glob import glob
from sparc import load_model
import pickle as pkl
from tqdm import tqdm

all_wavs = glob('./emg_data/**/*.wav', recursive=True)
coder = load_model("en+", device="cuda:0")

for wv in tqdm(all_wavs, total=len(all_wavs)):
    code = coder.encode(wv)
    out_name = wv.replace('.wav','_ema.pkl')
    with open(out_name, 'wb') as f:
        pkl.dump(code, f)
```

## Training
Below is an example script to train the model:
```sh
train.py --flagfile configs/config_example.txt
```

To train the model with only a subset of EMG electrodes, you can specifiy the selection using the following options. Note that indexing starts from $0$ in the code, whereas in the paper it starts from $1$. For instance, EMG electrode index 3 in the code corresponds to electrode 4 in the paper.
- `--remove_channels` specifies the indices of the EMG electrodes to exclude
- `--n_emg_ch` sets the total number of EMG electrodes.
```sh
train.py --flagfile configs/config_example.txt --remove_channels 0,4,6,7 --n_emg_ch 4
```

## Inference
You can run the inference script as shown below. This will output the predicted articulatory features.
```sh
inference.py --flagfile configs/config_example.txt --checkpoint_idx <epoch_idx>
```

To further synthesize speech waveforms from the predicted articulatory features, you can use the articulatory synthesis model from [SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding). Below is an example code snippet.
```python
from glob import glob
import os
from sparc import load_model
import pickle as pkl
from tqdm import tqdm
import soundfile as sf

out_dir = '<output_dir>'
all_emas = glob('<articulatory_feature_dir>/*.pkl', recursive=True)

coder = load_model("en+", device="cuda:0")

for wv in tqdm(all_emas, total=len(all_emas)):
    with open(wv, 'rb') as f:
        code = pkl.load(f)
    wav = coder.decode(**code)
    out_name = '<output_filename>'
    sf.write(out_name, wav, coder.sr)
```


## References
#### As mentioned in the paper, the implementation code is based on this [repository (https://github.com/dgaddy/silent_speech)](https://github.com/dgaddy/silent_speech).

