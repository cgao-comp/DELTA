import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from .htbi_block import HTBIBlock
from models.utils.config import OptConfig
import math

class ourModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--focal_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--temperature', type=float, default=0.007, help='temperature of contrastive learning loss')
        parser.add_argument('--Transformer_head', type=int, default=2, help='head of Transformer_head')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = []
        self.model_names = []
        self.netBackboneA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('BackboneA')
        self.netBackboneV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('BackboneV')
        self.netBackboneL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5)
        self.model_names.append('BackboneL')
        self.netLnBackboneA = nn.LayerNorm(opt.hidden_size)
        self.model_names.append('LnBackboneA')
        self.netLnBackboneV = nn.LayerNorm(opt.hidden_size)
        self.model_names.append('LnBackboneV')
        self.netLnBackboneL = nn.LayerNorm(opt.hidden_size)
        self.model_names.append('LnBackboneL')
        dfd_hidden = opt.hidden_size
        self.netDFD_encoder_inv = torch.nn.Linear(dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_encoder_inv')
        self.netDFD_encoder_sp_t = torch.nn.Linear(dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_encoder_sp_t')
        self.netDFD_encoder_sp_a = torch.nn.Linear(dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_encoder_sp_a')
        self.netDFD_encoder_sp_v = torch.nn.Linear(dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_encoder_sp_v')
        self.netDFD_decoder_t = torch.nn.Linear(2 * dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_decoder_t')
        self.netDFD_decoder_a = torch.nn.Linear(2 * dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_decoder_a')
        self.netDFD_decoder_v = torch.nn.Linear(2 * dfd_hidden, dfd_hidden)
        self.model_names.append('DFD_decoder_v')
        self.alpha_o = 0.1
        self.alpha_r = 0.01
        self.alpha_c = 0.01
        self.alpha_t = 0.1
        self.dfd_decay = 0.3
        self.loss_names.append('DFD')
        self.netIGA_proj_t = nn.Linear(opt.hidden_size, opt.emo_output_dim)
        self.model_names.append('IGA_proj_t')
        self.netIGA_proj_a = nn.Linear(opt.hidden_size, opt.emo_output_dim)
        self.model_names.append('IGA_proj_a')
        self.netIGA_proj_v = nn.Linear(opt.hidden_size, opt.emo_output_dim)
        self.model_names.append('IGA_proj_v')
        self.iga_eps = 1e-08
        self.iga_clamp_min, self.iga_clamp_max = (0.9, 1.1)
        self.iga_aux_weight = 0.1
        self.loss_names.append('IGA_aux')
        mdw_input_dim = 3 * opt.hidden_size
        mdw_hidden_dim = mdw_input_dim
        self.netMDW_mlp_emo = torch.nn.Sequential(torch.nn.Linear(mdw_input_dim, mdw_hidden_dim), torch.nn.ReLU(), torch.nn.Linear(mdw_hidden_dim, 3))
        self.model_names.append('MDW_mlp_emo')
        self.netMDW_mlp_int = torch.nn.Sequential(torch.nn.Linear(mdw_input_dim, mdw_hidden_dim), torch.nn.ReLU(), torch.nn.Linear(mdw_hidden_dim, 3))
        self.model_names.append('MDW_mlp_int')
        self.netHTBI = HTBIBlock(hidden_size=opt.hidden_size, num_layers=2, init_tokens=4, num_heads=int(opt.Transformer_head), ff_dim=opt.hidden_size * 4, dropout=opt.dropout_rate)
        self.model_names.append('HTBI')
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = 3 * opt.hidden_size
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')
        self.netIntC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.int_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('IntC')
        self.loss_names.append('int_CE')
        self.temperature = opt.temperature
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def load_pretrained_encoder(self, _opt):
        pass

    def post_process(self):
        pass

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.lexical = input['L_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        self.int_label = input['int_label'].to(self.device)

    def _log_tensor(self, t, name):
        log = getattr(self, 'logger', None)
        if log is None:
            return
        with torch.no_grad():
            s = t.shape
            m = t.float().mean().item()
            std = t.float().std().item()
            mn = t.float().min().item()
            mx = t.float().max().item()
            n = t.numel()
            norm = t.float().norm().item() / n ** 0.5 if n > 0 else 0.0
        log.info('  %s: shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f norm=%.4f' % (name, tuple(s), m, std, mn, mx, norm))

    def forward(self):
        log = getattr(self, 'logger', None)
        do_log = log is not None and getattr(self, '_current_epoch', 0) == 1 and (not getattr(self, '_forward_log_done', False))
        if do_log:
            self._forward_log_done = True
            log.info('[our_model] ========== forward 首 batch (epoch 1) 各模块 shape/统计 ==========')
        feat_A = self.netLnBackboneA(self.netBackboneA(self.acoustic))
        feat_V = self.netLnBackboneV(self.netBackboneV(self.visual))
        feat_L = self.netLnBackboneL(self.netBackboneL(self.lexical))
        if do_log:
            log.info('[our_model] --- Backbone 输出 ---')
            self._log_tensor(feat_A, 'feat_A')
            self._log_tensor(feat_V, 'feat_V')
            self._log_tensor(feat_L, 'feat_L')
        z_inv_t = self.netDFD_encoder_inv(feat_L)
        z_inv_a = self.netDFD_encoder_inv(feat_A)
        z_inv_v = self.netDFD_encoder_inv(feat_V)
        z_sp_t = self.netDFD_encoder_sp_t(feat_L)
        z_sp_a = self.netDFD_encoder_sp_a(feat_A)
        z_sp_v = self.netDFD_encoder_sp_v(feat_V)
        recon_t = self.netDFD_decoder_t(torch.cat([z_inv_t, z_sp_t], dim=-1))
        recon_a = self.netDFD_decoder_a(torch.cat([z_inv_a, z_sp_a], dim=-1))
        recon_v = self.netDFD_decoder_v(torch.cat([z_inv_v, z_sp_v], dim=-1))
        z_sp_t_hat = self.netDFD_encoder_sp_t(recon_t)
        z_sp_a_hat = self.netDFD_encoder_sp_a(recon_a)
        z_sp_v_hat = self.netDFD_encoder_sp_v(recon_v)
        cos_t = F.cosine_similarity(z_inv_t, z_sp_t, dim=-1)
        cos_a = F.cosine_similarity(z_inv_a, z_sp_a, dim=-1)
        cos_v = F.cosine_similarity(z_inv_v, z_sp_v, dim=-1)
        loss_orth = (cos_t ** 2).mean() + (cos_a ** 2).mean() + (cos_v ** 2).mean()
        loss_rec = F.mse_loss(recon_t, feat_L) + F.mse_loss(recon_a, feat_A) + F.mse_loss(recon_v, feat_V)
        loss_con = F.mse_loss(z_sp_t_hat, z_sp_t) + F.mse_loss(z_sp_a_hat, z_sp_a) + F.mse_loss(z_sp_v_hat, z_sp_v)

        def _triplet_loss(z_inv, labels, margin=0.2):
            z = F.normalize(z_inv, dim=-1)
            sim_mat = torch.matmul(z, z.t())
            labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            mask_pos = labels_equal - torch.eye(z.size(0), device=z.device)
            mask_neg = 1.0 - labels_equal
            sim_pos = (sim_mat * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-08)
            sim_neg = (sim_mat * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-08)
            loss = F.relu(sim_neg - sim_pos + margin)
            return loss.mean()
        with torch.no_grad():
            emo_labels = self.emo_label.view(-1)
        loss_trip = _triplet_loss(z_inv_t, emo_labels) + _triplet_loss(z_inv_a, emo_labels) + _triplet_loss(z_inv_v, emo_labels)
        dfd_raw = self.alpha_o * loss_orth + self.alpha_r * loss_rec + self.alpha_c * loss_con + self.alpha_t * loss_trip
        self.loss_DFD = dfd_raw * self.dfd_decay
        h_t = z_inv_t + z_sp_t
        h_a = z_inv_a + z_sp_a
        h_v = z_inv_v + z_sp_v
        if do_log:
            log.info('[our_model] --- DFD ---')
            self._log_tensor(z_inv_t, 'z_inv_t')
            self._log_tensor(z_sp_t, 'z_sp_t')
            self._log_tensor(recon_t, 'recon_t')
            self._log_tensor(h_t, 'h_t')
            self._log_tensor(h_a, 'h_a')
            self._log_tensor(h_v, 'h_v')
            log.info('  loss_orth=%.4f loss_rec=%.4f loss_con=%.4f loss_trip=%.4f loss_DFD=%.4f' % (loss_orth.item(), loss_rec.item(), loss_con.item(), loss_trip.item(), self.loss_DFD.item()))
        logits_iga_t = self.netIGA_proj_t(h_t.detach())
        logits_iga_a = self.netIGA_proj_a(h_a.detach())
        logits_iga_v = self.netIGA_proj_v(h_v.detach())
        emo_idx = self.emo_label.view(-1).long().clamp(0, logits_iga_t.size(1) - 1)
        prob_t = F.softmax(logits_iga_t, dim=-1).gather(1, emo_idx.unsqueeze(1)).squeeze(1)
        prob_a = F.softmax(logits_iga_a, dim=-1).gather(1, emo_idx.unsqueeze(1)).squeeze(1)
        prob_v = F.softmax(logits_iga_v, dim=-1).gather(1, emo_idx.unsqueeze(1)).squeeze(1)
        D_t = (1.0 - prob_t).abs()
        D_a = (1.0 - prob_a).abs()
        D_v = (1.0 - prob_v).abs()
        eps = self.iga_eps
        S_ta = (D_t - D_a).sum()
        S_tv = (D_t - D_v).sum()
        S_av = (D_a - D_v).sum()
        S_at = (D_a - D_t).sum()
        S_vt = (D_v - D_t).sum()
        S_va = (D_v - D_a).sum()
        num_ta = S_ta - (D_t - D_a)
        num_tv = S_tv - (D_t - D_v)
        num_at = S_at - (D_a - D_t)
        num_av = S_av - (D_a - D_v)
        num_vt = S_vt - (D_v - D_t)
        num_va = S_va - (D_v - D_a)
        W_t = 0.5 * (num_ta / (S_ta + eps) + num_tv / (S_tv + eps))
        W_a = 0.5 * (num_at / (S_at + eps) + num_av / (S_av + eps))
        W_v = 0.5 * (num_vt / (S_vt + eps) + num_va / (S_va + eps))
        W_t = torch.clamp(W_t, min=self.iga_clamp_min, max=self.iga_clamp_max)
        W_a = torch.clamp(W_a, min=self.iga_clamp_min, max=self.iga_clamp_max)
        W_v = torch.clamp(W_v, min=self.iga_clamp_min, max=self.iga_clamp_max)
        W_t = torch.where(torch.isfinite(W_t), W_t, torch.ones_like(W_t)).detach()
        W_a = torch.where(torch.isfinite(W_a), W_a, torch.ones_like(W_a)).detach()
        W_v = torch.where(torch.isfinite(W_v), W_v, torch.ones_like(W_v)).detach()
        self._iga_W_t = W_t.unsqueeze(1)
        self._iga_W_a = W_a.unsqueeze(1)
        self._iga_W_v = W_v.unsqueeze(1)
        if self.isTrain:
            h_t.register_hook(lambda g: g * self._iga_W_t)
            h_a.register_hook(lambda g: g * self._iga_W_a)
            h_v.register_hook(lambda g: g * self._iga_W_v)
        self.loss_IGA_t = self.criterion_ce(logits_iga_t, emo_idx)
        self.loss_IGA_a = self.criterion_ce(logits_iga_a, emo_idx)
        self.loss_IGA_v = self.criterion_ce(logits_iga_v, emo_idx)
        self.loss_IGA_aux = self.loss_IGA_t + self.loss_IGA_a + self.loss_IGA_v
        mdw_context = torch.cat([h_t, h_a, h_v], dim=-1)
        emo_scores = self.netMDW_mlp_emo(mdw_context)
        int_scores = self.netMDW_mlp_int(mdw_context)
        emo_weights = torch.sigmoid(emo_scores).unsqueeze(-1)
        int_weights = torch.sigmoid(int_scores).unsqueeze(-1)
        stack_h = torch.stack([h_t, h_a, h_v], dim=1)
        weighted_emo = emo_weights * stack_h
        weighted_int = int_weights * stack_h
        if do_log:
            log.info('[our_model] --- MDW (before LN) ---')
            self._log_tensor(emo_weights, 'emo_weights')
            self._log_tensor(int_weights, 'int_weights')
            self._log_tensor(weighted_emo, 'weighted_emo')
            self._log_tensor(weighted_int, 'weighted_int')
        weighted_emo = F.layer_norm(weighted_emo, weighted_emo.shape[2:])
        weighted_int = F.layer_norm(weighted_int, weighted_int.shape[2:])
        if do_log:
            log.info('[our_model] --- MDW (after LN) ---')
            self._log_tensor(weighted_emo, 'weighted_emo_ln')
            self._log_tensor(weighted_int, 'weighted_int_ln')
        self.h_emo_mdw = weighted_emo.sum(dim=1)
        self.h_int_mdw = weighted_int.sum(dim=1)
        emo_seq_out, int_seq_out = self.netHTBI(weighted_emo, weighted_int)
        emo_final_seq = emo_seq_out + weighted_emo
        int_final_seq = int_seq_out + weighted_int
        B_size = emo_final_seq.size(0)
        emo_final_feat = emo_final_seq.reshape(B_size, -1)
        int_final_feat = int_final_seq.reshape(B_size, -1)
        if do_log:
            log.info('[our_model] --- HTBI ---')
            self._log_tensor(emo_seq_out, 'emo_seq_out')
            self._log_tensor(int_seq_out, 'int_seq_out')
            self._log_tensor(emo_final_seq, 'emo_final_seq')
            self._log_tensor(int_final_seq, 'int_final_seq')
            self._log_tensor(emo_final_feat, 'emo_final_feat')
            self._log_tensor(int_final_feat, 'int_final_feat')
            log.info('[our_model] ========== end forward ==========')
        self.emo_logits, _ = self.netEmoC(emo_final_feat)
        self.int_logits, _ = self.netIntC(int_final_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)
        self.int_pred = F.softmax(self.int_logits, dim=-1)

    def backward(self):
        self.loss_emo_CE = self.ce_weight * self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_int_CE = self.ce_weight * self.criterion_ce(self.int_logits, self.int_label)
        loss = self.loss_emo_CE + self.loss_int_CE + self.loss_DFD
        loss = loss + self.iga_aux_weight * self.loss_IGA_aux
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self._current_epoch = epoch
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

class ActivateFun(torch.nn.Module):

    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
