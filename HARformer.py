from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos
from layers.Attention import WaveletAttention, FourierAttention, FullAttention
from layers.Attention import FourierAttention, FullAttention
from layers.RevIN import RevIN
import torch.nn.functional as F
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from pytorch_wavelets import DWT1D, IDWT1D
from exp.CKA import CKA, CudaCKA
class Model(nn.Module):
    """
    Transformer for seasonality, MLP for trend
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        #false
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dwt = DWT1D(wave='haar', mode='zero')
        self.idwt = IDWT1D(wave='haar', mode='zero')
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        #self.enc_seasonal_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                          configs.dropout)
        #self.dec_seasonal_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                          configs.dropout)
        # Encoder
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Time':
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)
        elif configs.version == 'Auto':
            enc_self_attention = AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
            dec_self_attention =AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
            dec_cross_attention =AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        #AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                         #               output_attention=False),
                        enc_self_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model),
                    AttentionLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Encoder
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

        self.revin_trend = RevIN(configs.enc_in).to(self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        cuda_cka = CudaCKA(self.device)
        # decomp init
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)  # cuda()
            #  x_enc.shape [batch_size,seq_len,feature]
            
        seasonal_enc, trend_enc = self.decomp(x_enc)
 
# 
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        #enc_out = self.enc_seasonal_embedding(enc_enc, x_mark_enc)
        #embedding
        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)
        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)
        ##
       
        ###  DWT 
        X=seasonal_dec
        X=X[:, -self.pred_len:, :]  
        
        enc_out_low,enc_out_high= self.dwt(enc_out)
        enc_out_1 = torch.cat((enc_out_low,enc_out_high[0]), dim=2)    
        enc_out_1, attn_e = self.seasonal_encoder(enc_out_1, attn_mask=enc_self_mask)

        

        ## IDWT
        enc_out_1_low,enc_out_1_high = torch.chunk(enc_out_1, chunks=2, dim=2)
    
        
        enc_out_1_high_1=[]
     

        enc_out_1_high_1.append(enc_out_1_high)

        enc_out=self.idwt((enc_out_1_low,enc_out_1_high_1))
        
        ###
    

        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)      
        seasonal_out = seasonal_out[:, -self.pred_len:, :]     
        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1) 
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        
        trend_enc = self.revin_trend(trend_enc, 'norm')
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, 'denorm')
        
  
        ###
        # final
        
        dec_out = trend_out + seasonal_ratio * seasonal_out
     
        
        return dec_out
