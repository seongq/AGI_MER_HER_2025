import torch
import torch.nn as nn
import torch.nn.functional as F

class TVAModel_Self(nn.Module): # trimodal self-attn model  
    def __init__(self, params):
        super(TVAModel_Self, self).__init__()
        self.params = params
        rnn = nn.LSTM if self.params.rnntype == "lstm" else nn.GRU



        if "t" in self.params.modals:
            self.text_encoder = rnn(input_size=self.params.txt_dim, hidden_size=self.params.txt_rnnsize,
                                    num_layers=self.params.txt_rnnnum, dropout=self.params.txt_rnndp, bidirectional=self.params.rnndir,
                                    batch_first=True)


            self.mha_t = nn.MultiheadAttention(embed_dim=2 * self.params.rnnsize, num_heads=self.params.txt_nh,
                                                    dropout=self.params.txt_dp, batch_first=True) 
            if (len(self.params.modals) > 1) and ('t' in self.params.aux_classifier):
                self.aux_classifier_unimodal_text = nn.Linear(in_features =  2 *self.params.rnnsize, out_features = self.params.output_dim)

        
        if "v" in self.params.modals:
            self.video_conv = nn.Conv1d(in_channels=32, out_channels=25, kernel_size=1) # 825 params
            self.video_encoder = rnn(input_size=self.params.vid_dim, hidden_size=self.params.vid_rnnsize,
                                    num_layers=self.params.vid_rnnnum, dropout=self.params.vid_rnndp, bidirectional=self.params.rnndir,
                                    batch_first=True)

            self.mha_v = nn.MultiheadAttention(embed_dim=2 * self.params.rnnsize, num_heads=self.params.vid_nh,
                                                dropout=self.params.vid_dp, batch_first=True)
            if (len(self.params.modals) > 1) and ('v' in self.params.aux_classifier):
                self.aux_classifier_unimodal_video = nn.Linear(in_features =  2 *self.params.rnnsize, out_features = self.params.output_dim)
        if "a" in self.params.modals:
            self.audio_conv = nn.Conv1d(in_channels=1000, out_channels=500, kernel_size=1)
            self.audio_encoder = rnn(input_size=self.params.aud_dim, hidden_size=self.params.aud_rnnsize,
                                    num_layers=self.params.aud_rnnnum, dropout=self.params.aud_rnndp, bidirectional=self.params.rnndir,
                                    batch_first=True)
        
        
            
            self.mha_a = nn.MultiheadAttention(embed_dim=2 * self.params.rnnsize, num_heads=self.params.aud_nh,
                                                dropout=self.params.aud_dp, batch_first=True)


            if (len(self.params.modals) > 1) and ('a' in self.params.aux_classifier):
                self.aux_classifier_unimodal_audio = nn.Linear(in_features =  2 *self.params.rnnsize, out_features = self.params.output_dim)

        ###original fusion, bimodal, trimodal
        if len(self.params.modals) > 1:
            self.concat_linear = nn.Linear(in_features=2 * 2 * self.params.rnnsize, out_features= self.params.rnnsize)
            self.classifier = nn.Linear(in_features= self.params.rnnsize, out_features=self.params.output_dim)
        elif len(self.params.modals) == 1:
            self.jungri_linear = nn.Linear(in_features=2*self.params.rnnsize, out_features= self.params.rnnsize)
            self.classifier_unimodal = nn.Linear(in_features =  self.params.rnnsize, out_features = self.params.output_dim)

    def forward(self, x_txt, x_vid, x_audio, emb_dp=0.25):
        # text branch

        if "t" in self.params.modals:
            x_txt = F.dropout(x_txt, p=emb_dp, training=self.training) 
            x_txt, h = self.text_encoder(x_txt) 
            x_txt, _ = self.mha_t(x_txt,x_txt,x_txt) 
            x_txt2 = torch.mean(x_txt, dim=1)
       
        if "v" in self.params.modals:
            # video branch
            x_vid = self.video_conv(x_vid)
            x_vid, h = self.video_encoder(x_vid)
            x_vid, _ = self.mha_v(x_vid, x_vid, x_vid) 
            x_vid2 = torch.mean(x_vid, dim=1)
       


        if "a" in self.params.modals:
            # audio branch
            x_audio = self.audio_conv(x_audio)
            x_audio, h = self.audio_encoder(x_audio)
            x_audio, _ = self.mha_a(x_audio, x_audio, x_audio)
            x_audio2 = torch.mean(x_audio, dim=1)
      
      
        ###original version & fusion

        if self.params.modals == "tva":
            x_tva1 = torch.stack((x_txt2, x_vid2, x_audio2), dim=1) 
            x_tva1_mean, x_tva1_std = torch.std_mean(x_tva1, dim=1)
            x_tva = torch.cat((x_tva1_mean, x_tva1_std), dim=1) 
            x_tva = self.concat_linear(x_tva)
            y = self.classifier(x_tva) # [32, 7]
            
            
            x_aux_t = self.aux_classifier_unimodal_text(x_txt2)
            x_aux_v = self.aux_classifier_unimodal_video(x_vid2)
            x_aux_a = self.aux_classifier_unimodal_audio(x_audio2)
            
            
            return y, x_tva, x_aux_t, x_aux_v, x_aux_a

        if self.params.modals == "tv":
            x_tv1 = torch.stack((x_txt2, x_vid2), dim=1) 
            x_tv1_mean, x_tv1_std = torch.std_mean(x_tv1, dim=1)
            x_tv = torch.cat((x_tv1_mean, x_tv1_std), dim=1) 
            x_tv = self.concat_linear(x_tv)
            y = self.classifier(x_tv) # [32, 7]
            
            
            x_aux_t = self.aux_classifier_unimodal_text(x_txt2)
            x_aux_v = self.aux_classifier_unimodal_video(x_vid2)

            
            
            return y, x_tv, x_aux_t, x_aux_v

        if self.params.modals == "ta":
            x_ta1 = torch.stack((x_txt2, x_audio2), dim=1) 
            x_ta1_mean, x_ta1_std = torch.std_mean(x_ta1, dim=1)
            x_ta = torch.cat((x_ta1_mean, x_ta1_std), dim=1) 
            x_ta = self.concat_linear(x_ta)
            y = self.classifier(x_ta) # [32, 7]
            
            x_aux_t = self.aux_classifier_unimodal_text(x_txt2)
            x_aux_a = self.aux_classifier_unimodal_audio(x_audio2)
            return y, x_ta, x_aux_t, x_aux_a

        if self.params.modals == "va":
            x_va1 = torch.stack((x_vid2, x_audio2), dim=1) 
            x_va1_mean, x_va1_std = torch.std_mean(x_va1, dim=1)
            x_va = torch.cat((x_va1_mean, x_va1_std), dim=1) 
            x_va = self.concat_linear(x_va)
            y = self.classifier(x_va) # [32, 7]
            
            
            x_aux_v = self.aux_classifier_unimodal_video(x_vid2)
            x_aux_a = self.aux_classifier_unimodal_audio(x_audio2)
            return y, x_va, x_aux_v, x_aux_a


        if self.params.modals == "v":
            x_v = x_vid2 
            x_v = self.jungri_linear(x_v)
            y = self.classifier_unimodal(x_v) # [32, 7]
            return y, x_v

        if self.params.modals == "t":
            x_t = x_txt2 
            x_t = self.jungri_linear(x_t)
            y = self.classifier_unimodal(x_t) # [32, 7]
            return y, x_t

        if self.params.modals == "a":
            x_a = x_audio2
            x_a = self.jungri_linear(x_a)
            y = self.classifier_unimodal(x_a)
            return y, x_a
    
    
class TVAModel_Cross(nn.Module): # trimodal cross-attn model
    def __init__(self, params):
        super(TVAModel_Cross, self).__init__()
        rnn = nn.LSTM if params.rnntype == "lstm" else nn.GRU
        self.text_encoder = rnn(input_size=params.txt_dim, hidden_size=params.txt_rnnsize,
                                num_layers=params.txt_rnnnum, dropout=params.txt_rnndp, bidirectional=params.rnndir,
                                batch_first=True)
        self.video_conv = nn.Conv1d(in_channels=32, out_channels=25, kernel_size=1) # 825 params
        self.video_encoder = rnn(input_size=params.vid_dim, hidden_size=params.vid_rnnsize,
                                 num_layers=params.vid_rnnnum, dropout=params.vid_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)
        self.audio_conv = nn.Conv1d(in_channels=1000, out_channels=500, kernel_size=1)
        self.audio_encoder = rnn(input_size=params.aud_dim, hidden_size=params.aud_rnnsize,
                                 num_layers=params.aud_rnnnum, dropout=params.aud_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)
        if params.rnndir:
            self.mha_v_t = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.vt_nh,
                                               dropout=params.vt_dp, batch_first=True)
            self.mha_a_t = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.at_nh,
                                                 dropout=params.at_dp, batch_first=True)
            self.mha_t_v = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.tv_nh,
                                                 dropout=params.tv_dp, batch_first=True)
            self.mha_a_v = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.av_nh,
                                                 dropout=params.av_dp, batch_first=True)
            self.mha_t_a = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.ta_nh,
                                                 dropout=params.ta_dp, batch_first=True)
            self.mha_v_a = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.va_nh,
                                                 dropout=params.va_dp, batch_first=True)
            self.concat_linear = nn.Linear(in_features=2 * 2 * params.rnnsize, out_features= params.rnnsize)
            self.classifier = nn.Linear(in_features= params.rnnsize, out_features=params.output_dim)

    def forward(self, x_txt, x_vid, x_mfcc, emb_dp=0.25):
        # text branch
        x_txt = F.dropout(x_txt, p=emb_dp, training=self.training) 
        x_txt, h = self.text_encoder(x_txt) 
        # video branch
        x_vid = self.video_conv(x_vid)
        x_vid, h = self.video_encoder(x_vid)
        # audio branch
        x_mfcc = self.audio_conv(x_mfcc)
        x_mfcc, h = self.audio_encoder(x_mfcc)
        ##### V,A -> T
        # video to text
        x_v2t, _ = self.mha_v_t(x_txt, x_vid, x_vid) 
        x_v2t = torch.mean(x_v2t, dim=1)
        # audio to text
        x_a2t, _ = self.mha_a_t(x_txt, x_mfcc, x_mfcc)
        x_a2t = torch.mean(x_a2t, dim=1)
        # addition
        ####### T,A -> V
        # text to video
        x_t2v, _ = self.mha_t_v(x_vid, x_txt, x_txt) 
        x_t2v = torch.mean(x_t2v, dim=1)
        # audio to video
        x_a2v, _ = self.mha_a_v(x_vid, x_mfcc, x_mfcc)
        x_a2v = torch.mean(x_a2v, dim=1)
        # addition
        ####### T,V -> A
        # text to audio
        x_t2a, _ = self.mha_t_a(x_mfcc, x_txt, x_txt)
        x_t2a = torch.mean(x_t2a, dim=1)
        # video to audio
        x_v2a, _ = self.mha_v_a(x_mfcc, x_vid, x_vid)
        x_v2a = torch.mean(x_v2a, dim=1)

        x_tva2 = torch.stack((x_a2t, x_v2t, x_t2v, x_a2v, x_t2a, x_v2a), dim=1) 
        x_tva2_mean, x_tva2_std = torch.std_mean(x_tva2, dim=1)
        x_tva2 = torch.cat((x_tva2_mean, x_tva2_std), dim=1) 
        x_tva = x_tva2
        x_tva = self.concat_linear(x_tva)
        y = self.classifier(x_tva) # [32, 7]
        return y, x_tva

