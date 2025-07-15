# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
from torchvision.models.optical_flow import raft_small
from detr.models.flow_backbone import FlowBackbone  # 绝对路径导入 stevez
import IPython
e = IPython.embed
import torch.nn.functional as F
# stevez
class AttentionFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        # 初始化（optional，但推荐）
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attn = self.attention(x)
        out = x * (1 + attn)  # residual attention，保证信息流
        out = self.out_conv(out)
        return out

class OpticalFlowExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.flownet = FlowNetSimple(pretrained=True)
        # self.flownet = raft_small(pretrained=True).eval().half() # half precision
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.flownet = raft_small(pretrained=True).eval().float().to(device) # or half(float 16)

    def forward(self, prev_imgs, current_imgs):
        # prev_imgs: [batch, num_cam, 3, H, W]
        # current_imgs: [batch, num_cam, 3, H, W]
        # Returns optical flow [batch, num_cam, 2, H, W]
        flows = []
        
        for cam_id in range(prev_imgs.shape[1]):
            flow = self.flownet(prev_imgs[:, cam_id], current_imgs[:, cam_id])[-1]
            flows.append(flow)
        return torch.stack(flows, dim=1)
        

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        self.flow_extractor = OpticalFlowExtractor() # stevez
        self.flow_extractor = self.flow_extractor.float() # stevez
        hidden_dim = transformer.d_model
        # print(f"hidden_dim (transformer.d_model): {hidden_dim}")
        # print(F"action_dim: {action_dim} (for action prediction)")
        self.flow_backbones = nn.ModuleList([
            FlowBackbone(in_channels=2, hidden_dim=hidden_dim, debug=True)  # 输入通道为2（光流场的x,y分量）
            for _ in camera_names
        ])
        # stevez
        self.alpha_conv = nn.Sequential(
                nn.Conv2d(2 * 512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Sigmoid()
            )
        self.input_proj_flow = nn.Conv2d(self.flow_backbones[0].num_channels, hidden_dim, kernel_size=1)
        # self.fusion_conv = AttentionFusion(1024, hidden_dim=1)
        self.fusion_conv = AttentionFusion(1024, hidden_dim=256)

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        probs = binaries = mu = logvar = None
        
        is_training = actions is not None  # Training or evaluation phase
        
        if is_training:
            # Project the action sequence to the embedding dimension
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            # Use action_embed batch size as the true batch size in case of mismatch
            actual_bs = action_embed.shape[0]
        else:
            # In the evaluation phase, if there are no actions, create a placeholder action_embed
            hidden_dim = self.encoder_action_proj.out_features
            action_embed = torch.zeros(bs, 0, hidden_dim, device=qpos.device)  # (bs, 0, hidden_dim)
            actual_bs = bs
        
        # Fix qpos_embed to ensure correct batch size - use actual_bs
        qpos_embed = self.encoder_joint_proj(qpos)  # Shape [bs, 512]
        if qpos_embed.shape[0] != actual_bs:
            # If there's a batch size mismatch, expand or slice qpos_embed to match
            if qpos_embed.shape[0] == 1 and actual_bs > 1:
                qpos_embed = qpos_embed.expand(actual_bs, -1)  # Expand to match batch size
            elif qpos_embed.shape[0] > actual_bs:
                qpos_embed = qpos_embed[:actual_bs]  # Slice to match batch size
        qpos_embed = qpos_embed.unsqueeze(1)  # Shape [actual_bs, 1, 512]
        
        # Fix cls_embed to ensure correct batch size - use actual_bs
        cls_embed = self.cls_embed.weight  # Shape [1, 512]
        cls_embed = cls_embed.unsqueeze(0)  # Shape [1, 1, 512]
        cls_embed = cls_embed.expand(actual_bs, -1, -1)  # Shape [actual_bs, 1, 512]
        
        # Concatenate tensors along sequence dimension
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # [actual_bs, 102, 512] or [actual_bs, 2, 512]
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, actual_bs, hidden_dim)
        
        cls_joint_is_pad = torch.full((actual_bs, 2), False, device=qpos.device)  # False: not a padding
        
        if is_training:
            if is_pad is not None:
                is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (actual_bs, seq+2)
            else:
                # If is_pad is None, create a tensor with appropriate dimensions
                seq_length = action_embed.shape[1]
                is_pad = torch.cat([cls_joint_is_pad, torch.full((actual_bs, seq_length), False, device=qpos.device)], dim=1)
        else:
            is_pad = cls_joint_is_pad
        
        # Obtain position embedding
        if is_training:
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
        else:
            # In the evaluation phase, the position embedding only needs the cls and qpos parts
            pos_embed = self.pos_table[:, :2].clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (2, 1, hidden_dim)
        
        # Query the model
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
        encoder_output = encoder_output[0]  # Take only the cls output
        latent_info = self.latent_proj(encoder_output)
        
        if self.vq:
            logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
            probs = torch.softmax(logits, dim=-1)
            binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
            binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
            probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
            straigt_through = binaries_flat - probs_flat.detach() + probs_flat
            latent_input = self.latent_out_proj(straigt_through)
            mu = logvar = None
        else:
            probs = binaries = None
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        
        return latent_input, probs, binaries, mu, logvar
   
    # stevez
    def forward(self, qpos, cur_image, env_state, actions=None, is_pad=None, vq_sample=None, prev_image=None):
        device = qpos.device  # 获取模型当前设备
        if actions is not None:
            actions = actions.float().to(device)
        else:
            actions = torch.zeros(8, 100, 16, device=device)
        
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        if self.backbones and self.flow_backbones is not None:
      
            with torch.no_grad():
                if prev_image is not None:
                    prev_image = prev_image.float()
                    cur_image = cur_image.float()
                    flow = self.flow_extractor(prev_image, cur_image)  # [bs, num_cam, 2, H, W]
                    max_flow = 20.0
                    flow = torch.clamp(flow, -max_flow, max_flow) / max_flow

            
         
            all_features = []
            all_positions = []
            for cam_id in range(len(self.camera_names)):
                if prev_image is not None:
                    flow_feat, flow_pos = self.flow_backbones[cam_id](flow[:, cam_id])
                    flow_feat = self.input_proj_flow(flow_feat[0])  # [bs, hidden_dim, H, W]
                    flow_pos = flow_pos[0]  # [bs, hidden_dim, H, W]
                    flow_pos = flow_pos[0:1]  # 取第一个 batch，对齐到 img_pos
                    flow_pos = F.interpolate(flow_pos, size=(15, 20), mode='bilinear', align_corners=False) #do resizing to fit 15x20 of img_pos
                    flow_feat = F.interpolate(flow_feat, size=(15, 20), mode='bilinear', align_corners=False) # same to fit img_feat
              
                img_feat, img_pos = self.backbones[cam_id](cur_image[:, cam_id])
                img_feat = self.input_proj(img_feat[0])  # [bs, hidden_dim, H, W]
                img_pos = img_pos[0]  
               
                if prev_image is not None:
                   
                    fused_feat = torch.cat([img_feat, flow_feat], dim=1)  # [bs, hidden_dim*2, H, W]
                    fused_feat = self.fusion_conv(fused_feat)  # [bs, hidden_dim, H, W]
                    # print(self.fusion_conv)
                    all_features.append(fused_feat)
                    
                    
                    if flow_pos is not None and img_pos is not None:
                        print(f"img_pos_shape: {img_pos.shape}")
                        print(f"flow_pos_shape: {flow_pos.shape}")
                        alpha = self.alpha_conv(torch.cat([img_pos, flow_pos], dim=1))
                        fused_pos = alpha * img_pos + (1 - alpha) * flow_pos
                        print(f"[DEBUG] fused_feat: min={fused_feat.min().item():.3f}, max={fused_feat.max().item():.3f}, mean={fused_feat.mean().item():.3f}, std={fused_feat.std().item():.3f}")
                        print(f"[DEBUG] fused_pos: min={fused_pos.min().item():.3f}, max={fused_pos.max().item():.3f}, mean={fused_pos.mean().item():.3f}, std={fused_pos.std().item():.3f}")

                    else:
                        fused_pos = img_pos    

                    all_positions.append(fused_pos)
                else:
                    all_features.append(img_feat)
                    all_positions.append(img_pos)
                    
            src = torch.cat(all_features, dim=3)  # [bs, hidden_dim, H, W*num_cam]
            pos = torch.cat(all_positions, dim=3)  # [bs, hidden_dim, H, W*num_cam]
        
           
            proprio_input = self.input_proj_robot_state(qpos)
            
           
            hs = self.transformer(
                src=src,
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pos,
                latent_input=latent_input,
                proprio_input=proprio_input,
                additional_pos_embed=self.additional_pos_embed.weight
            )[0]
        else:
           
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

  
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model