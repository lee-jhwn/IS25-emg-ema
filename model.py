import random
from torch import nn
import torch.nn.functional as F
from modules import TransformerEncoderLayer, ResBlock
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('model_size', 768, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')
flags.DEFINE_list('stride_sizes', [2, 2, 2], 'strides')
flags.DEFINE_integer('n_emg_ch', 8, 'number of emg channels')

class EMG2EMA(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(FLAGS.n_emg_ch, FLAGS.model_size, int(FLAGS.stride_sizes[0])),
            ResBlock(FLAGS.model_size, FLAGS.model_size, int(FLAGS.stride_sizes[1])),
            ResBlock(FLAGS.model_size, FLAGS.model_size, int(FLAGS.stride_sizes[2])),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)



        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

        self.pitch_proj = nn.Linear(FLAGS.model_size, 1)
        self.loudness_proj = nn.Linear(FLAGS.model_size, 1)
            
  
    def forward(self, x_feat, x_raw, session_ids):
        # x shape is (batch, time, electrode)


        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0

        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        x = x.transpose(0,1) # put time first
        x = self.transformer(x)
        x = x.transpose(0,1)

        ema_out, pitch_out, loudness_out, aux_out = self.w_out(x), self.pitch_proj(x), self.loudness_proj(x), self.w_aux(x)

        if self.has_aux_out:
            return ema_out, pitch_out, loudness_out, aux_out
        else:
            return self.w_out(x)

