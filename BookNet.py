from extractor import BasicEncoder
from position_encoding import build_position_encoding

import argparse
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional



class attnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.ModuleList([copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout)) for i in range(2)])
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)])

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_list = nn.ModuleList([copy.deepcopy(nn.Dropout(dropout)) for i in range(2)])
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        for memory, multihead_attn, norm2, dropout2, m_pos in zip(memory_list, self.multihead_attn_list, self.norm2_list, self.dropout2_list, memory_pos):
            tgt2 = multihead_attn(query=self.with_pos_embed(tgt, pos),
                                       key=self.with_pos_embed(memory, m_pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_list, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos)
        return self.forward_post(tgt, memory_list, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed, query_shape=None):
        device = imgf.device
        bs, c, h, w = imgf.shape

        pos = self.position_embedding((bs, h, w), device)
        pos = pos.flatten(2).permute(2, 0, 1)

        query_len = query_embed.shape[0]
        if query_shape is not None:
            query_h, query_w = query_shape
        elif query_len == 648:
            query_h, query_w = 36, 18
        elif query_len == 1296:
            query_h, query_w = 36, 36
        else:
            query_h, query_w = h, w

        query_pos = self.position_embedding((bs, query_h, query_w), device)
        query_pos = query_pos.flatten(2).permute(2, 0, 1)

        imgf = imgf.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=query_pos, memory_pos=[pos, pos])

        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, query_h, query_w)

        return query_embed


class TransEncoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf):
        device = imgf.device
        bs, c, h, w = imgf.shape

        pos = self.position_embedding((bs, h, w), device)
        pos = pos.flatten(2).permute(2, 0, 1)

        imgf = imgf.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])

        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)
        return imgf


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = .25 * self.mask(imgf)
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, key_value_feat):
        B, C, H, W = query_feat.shape

        query = query_feat.flatten(2).permute(2, 0, 1)
        key = key_value_feat.flatten(2).permute(2, 0, 1)
        value = key_value_feat.flatten(2).permute(2, 0, 1)

        attn_output, attn_weights = self.multihead_attn(query, key, value)

        enhanced_feat = query + self.dropout(attn_output)
        enhanced_feat = self.norm(enhanced_feat)

        enhanced_feat = enhanced_feat.permute(1, 2, 0).reshape(B, C, H, W)

        return enhanced_feat, attn_weights


class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim=256):
        super(FeatureFusion, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
        )

    def forward(self, fmap_left, fmap_right, coords1):
        fused_feat = torch.cat([fmap_left, fmap_right], dim=3)

        fused_feat = self.fusion_conv(fused_feat)

        mask = 0.25 * self.mask(fused_feat)
        dflow = self.flow_head(fused_feat)
        coords_full = coords1 + dflow

        return mask, coords_full


class BookNet(nn.Module):
    def __init__(self, num_attn_layers):
        super(BookNet, self).__init__()
        self.num_attn_layers = num_attn_layers

        self.hidden_dim = hdim = 256

        self.backbone = BasicEncoder(output_dim=hdim, norm_fn='instance')

        self.encoder = TransEncoder(self.num_attn_layers, hidden_dim=hdim)
        self.decoder_left_1 = TransDecoder(self.num_attn_layers//2, hidden_dim=hdim)
        self.decoder_right_1 = TransDecoder(self.num_attn_layers//2, hidden_dim=hdim)

        self.cross_attn_left = CrossAttention(hidden_dim=hdim)
        self.cross_attn_right = CrossAttention(hidden_dim=hdim)

        self.decoder_left_2 = TransDecoder(self.num_attn_layers//2, hidden_dim=hdim)
        self.decoder_right_2 = TransDecoder(self.num_attn_layers//2, hidden_dim=hdim)

        num_queries_left = 36 * 18
        num_queries_right = 36 * 18

        query_embed = nn.Embedding(num_queries_left + num_queries_right, self.hidden_dim)
        embed_weight = query_embed.weight
        self.query_embed_left = nn.Parameter(embed_weight[:num_queries_left])
        self.query_embed_right = nn.Parameter(embed_weight[num_queries_left:])

        self.update_block_left = UpdateBlock(self.hidden_dim)
        self.update_block_right = UpdateBlock(self.hidden_dim)

        self.feature_fusion = FeatureFusion(self.hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coodslar, coords0, coords1

    def initialize_flow_half_width(self, img):
        N, C, H, W = img.shape
        coodslar_half = coords_grid(N, H, W // 2).to(img.device)
        coords0_half = coords_grid(N, H // 8, W // 16).to(img.device)
        coords1_half = coords_grid(N, H // 8, W // 16).to(img.device)
        return coodslar_half, coords0_half, coords1_half

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image):
        fmap = self.backbone(image)
        fmap = torch.relu(fmap)
        fmap = self.encoder(fmap)

        bs = fmap.size(0)
        query_left = self.query_embed_left.unsqueeze(1).repeat(1, bs, 1)
        query_right = self.query_embed_right.unsqueeze(1).repeat(1, bs, 1)

        fmap_left = self.decoder_left_1(fmap, query_left, query_shape=(36, 18))
        fmap_right = self.decoder_right_1(fmap, query_right, query_shape=(36, 18))

        enhanced_left, attn_weights_l2r = self.cross_attn_left(fmap_left, fmap_right)
        enhanced_right, attn_weights_r2l = self.cross_attn_right(fmap_right, fmap_left)

        enhanced_left_query = enhanced_left.flatten(2).permute(2, 0, 1)
        enhanced_right_query = enhanced_right.flatten(2).permute(2, 0, 1)

        fmap_left_2 = self.decoder_left_2(fmap, enhanced_left_query, query_shape=(36, 18))
        fmap_right_2 = self.decoder_right_2(fmap, enhanced_right_query, query_shape=(36, 18))

        coords_large_half, coords0_half, coords1_half = self.initialize_flow_half_width(image)
        coords1_half = coords1_half.detach()

        mask_left, coords_left = self.update_block_left(fmap_left_2, coords1_half)
        mask_right, coords_right = self.update_block_right(fmap_right_2, coords1_half)

        flow_left = self.upsample_flow(coords_left - coords0_half, mask_left)
        flow_right = self.upsample_flow(coords_right - coords0_half, mask_right)

        bm_left = coords_large_half + flow_left
        bm_right = coords_large_half + flow_right

        coords_large_full, coords0_full, coords1_full = self.initialize_flow(image)
        coords1_full = coords1_full.detach()

        mask_full, coords_full = self.feature_fusion(fmap_left_2, fmap_right_2, coords1_full)
        flow_full = self.upsample_flow(coords_full - coords0_full, mask_full)
        bm_full = coords_large_full + flow_full

        return bm_left, bm_right, bm_full


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BookNet(num_attn_layers=4).to(device)

    batch_size = 2
    test_input = torch.randn(batch_size, 3, 288, 288).to(device)

    bm_left, bm_right, bm_full = model(test_input)

    print("Input shape:", test_input.shape)
    print("Left BM shape:", bm_left.shape)
    print("Right BM shape:", bm_right.shape)
    print("Full BM shape:", bm_full.shape)
    print(get_parameter_number(model))
