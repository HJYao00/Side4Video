from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from .factory import create_model_and_transforms

from modules.until_module import AllGather, CrossEn

allgather = AllGather.apply


WETGHT_PATH = {
            "EVA02-CLIP-L-14": './clip-pretrain/EVA02_CLIP_L_psz14_s4B.pt',
            "EVA02-CLIP-L-14-336": './clip-pretrain/EVA02_CLIP_L_336_psz14_s6B.pt',
            "EVA02-CLIP-bigE-14":'./clip-pretrain/EVA02_CLIP_E_psz14_s4B.pt',
            "EVA02-CLIP-bigE-14-plus":'./clip_pretrain/EVA02_CLIP_E_psz14_plus_s9B.pt'
        }

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4CLIP(nn.Module):
    def __init__(self, args):
        super(CLIP4CLIP, self).__init__()
        self.task_config = args
        self.clip, _, preprocess = create_model_and_transforms(args.pretrained_clip_name,\
                             pretrained=WETGHT_PATH[args.pretrained_clip_name], force_custom_clip=True, T=args.max_frames, side_dim=args.side_dim)
        
        self.clip.float()

        self.loose_type = True
        # if self._stage_one and check_attr('loose_type', self.task_config):
        #     self.loose_type = True
            # show_log(task_config, "Test retrieval by loose type.")

        if self.task_config.interaction == 'wti':
            self.cdcr_alpha1=0.11
            self.cdcr_alpha2=0.0
            self.cdcr_lambda=0.001
            embed_dim = 1024
            self.text_weight_fc = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
                    nn.Linear(embed_dim, 1))
            self.video_weight_fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, 1))
        self.loss_fct = CrossEn()

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)   # float32

        if self.training:
            loss = 0.
            if self.task_config.interaction == 'wti':
                sim_matrix1, sim_matrix2, cdcr_loss = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                        shaped=True)
                sim_loss = (self.loss_fct(sim_matrix1) + self.loss_fct(sim_matrix2)) / 2.0
                loss = sim_loss + cdcr_loss * self.cdcr_lambda
            else:
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                        shaped=True)    # float16
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                sim_loss = (sim_loss1 + sim_loss2) / 2
                loss += sim_loss

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        if self.task_config.interaction == 'wti':
            sequence_hidden = self.clip.encode_text(input_ids, return_all_features=True)[1].float()
        else:
            sequence_hidden = self.clip.encode_text(input_ids).float()
        # sequence_hidden = self.clip.encode_text(input_ids).float() 

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video).float()
        if self.task_config.interaction != 'wti':
            visual_hidden = visual_hidden.mean(1)
        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output
    
    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        # sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous() # my zhushi

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * sequence_output @ visual_output.T # torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits



    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=None):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self.task_config.interaction == 'wti':
            t2v_logits, v2t_logits, cdcr_loss = self.wti_interaction(sequence_output, visual_output, attention_mask, video_mask)
            return t2v_logits, v2t_logits, cdcr_loss

        contrastive_direction = ()
        retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask)

        return retrieve_logits, contrastive_direction


    def wti_interaction(self, text_feat, video_feat, text_mask, video_mask):
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = allgather(text_feat, self.task_config)
            video_feat = allgather(video_feat, self.task_config)
            text_mask = allgather(text_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            torch.distributed.barrier()  # force sync

        # print(1, text_feat.shape)
        text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(video_feat).squeeze(2) # B x N_v x D -> B x N_v
        video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        # text_sum = text_mask.sum(-1)
        # video_sum = video_mask.sum(-1)

        # max for video token
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits
            
                # selecet max
            if self.task_config.cdcr:
                max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]), torch.arange(max_idx1.shape[1])]
                max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]), torch.arange(max_idx2.shape[1])]

                max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).repeat_interleave(max_idx2.shape[1]),
                                        max_idx2.flatten()].squeeze(1)
                max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1]),
                                        max_idx1.flatten()].squeeze(1)

                t_feat = text_feat.reshape(-1, text_feat.shape[-1])
                t_mask = text_mask.flatten().type(torch.bool)
                v_feat = video_feat.reshape(-1, video_feat.shape[-1])
                v_mask = video_mask.flatten().type(torch.bool)
                t_feat = t_feat[t_mask]
                v_feat = v_feat[v_mask]
                max_t_feat = max_t_feat[v_mask]
                max_v_feat = max_v_feat[t_mask]
                text_weight = text_weight.flatten()[t_mask]
                video_weight = video_weight.flatten()[v_mask]

                z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)  # (BxN_t)xD
                z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(0)  # (BxN_t)xD

                x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)  # (BxN_v)xD
                x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(0)  # (BxN_v)xD

                # cross-correlation matrix
                N, D = z_a_norm.shape
                B = text_feat.shape[0]
                c1 = torch.einsum("acd,a->cd", torch.einsum('ac,ad->acd', z_a_norm, z_b_norm),
                                    text_weight) / B  # DxD
                c2 = torch.einsum("acd,a->cd", torch.einsum('ac,ad->acd', x_a_norm, x_b_norm),
                                    video_weight) / B  # DxD
                c = (c1 + c2) / 2.0
                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.cdcr_alpha1 + off_diag * self.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            else:
                return retrieve_logits, retrieve_logits.T, 0.0 
        else:
            return retrieve_logits, retrieve_logits.T, 0.0
        