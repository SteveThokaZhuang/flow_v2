# 动过的代码
1. imitate episodes
2. **detr_vae**
3. **transformer**
4. policy
# 已知问题
success rate 和 average return rate没有返回
# 具体改动
## imitate episodes
1. 读取prev_image和cur_image以及prev_image的cuda化，把prev作为新参数传入policy
``` def forward_pass(data, policy):
    global prev_image_ulti
    image_data, qpos_data, action_data, is_pad = data
    prev_image = prev_image_ulti.cuda() if prev_image_ulti is not None else None # stevez
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    vq_sample = None
    # vq_sample = vq_sample.cuda()
    prev_image_ulti = image_data
    return policy(qpos_data, image_data, action_data, is_pad, vq_sample, prev_image) # TODO remove None
```
## policy (ACTpolicy)
1. 在 policy call中调用模型，**有可能是prev没有normalize**
```a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample, prev_image) #prev_image flow stevez```
## detrvae
1. forward接收prev_image并提取feature，**并和image feature融合**
```def forward(self, qpos, cur_image, env_state, actions=None, is_pad=None, vq_sample=None, prev_image=None):
        # print(f"action in forward: {actions.shape}")
        # print(f"action is :{actions}")
        device = qpos.device  # 获取模型当前设备
        if actions is not None:
            actions = actions.float().to(device)
        else:
            actions = torch.zeros(8, 100, 16, device=device)
        # print(f"actions shape : {actions.shape}")
        # actions = actions.float()
        # print(f"actions:{actions}")

        
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        if self.backbones and self.flow_backbones is not None:
      
            with torch.no_grad():
                if prev_image is not None:
                    prev_image = prev_image.float()
                    cur_image = cur_image.float()
                    flow = self.flow_extractor(prev_image, cur_image)  # [bs, num_cam, 2, H, W]
                    max_flow = 20.0  # 视你的场景而定，通常 10~40 像素
                    flow = torch.clamp(flow, -max_flow, max_flow) / max_flow

            
         
            all_features = []
            all_positions = []
            for cam_id in range(len(self.camera_names)):
                # 光流分支
                if prev_image is not None:
                    flow_feat, flow_pos = self.flow_backbones[cam_id](flow[:, cam_id])
                    flow_feat = self.input_proj_flow(flow_feat[0])  # [bs, hidden_dim, H, W]
                    flow_pos = flow_pos[0]  # [bs, hidden_dim, H, W]
                    flow_feat = F.interpolate(flow_feat, size=(15, 20), mode='bilinear', align_corners=False)

                    # print(f"flow_feat shape: {flow_feat.shape}")  # 例如 [bs, hidden_dim, 60, 60]

                # 图像分支
                img_feat, img_pos = self.backbones[cam_id](cur_image[:, cam_id])
                img_feat = self.input_proj(img_feat[0])  # [bs, hidden_dim, H, W]
                img_pos = img_pos[0]  # 确保img_pos是张量 [bs, hidden_dim, H, W]
                # print(f"img_feat shape: {img_feat.shape}")  # 例如 [bs, hidden_dim, 15, 15]
                
                # 特征融合
                if prev_image is not None:
                    # todo: 加权融合
                    # print(f"img_feat: {img_feat}")
                    # print(f"flow_feat: {flow_feat}")
                    fused_feat = torch.cat([img_feat, flow_feat], dim=1)  # [bs, hidden_dim*2, H, W]
                    fused_feat = self.fusion_conv(fused_feat)  # [bs, hidden_dim, H, W]
                    # print(self.fusion_conv)
                    all_features.append(fused_feat)
                    
                    # 关键修复：融合位置编码（示例：简单平均）
                    # if img_pos is not None:
                        # fused_pos = (img_pos + flow_pos) / 2  # [bs, hidden_dim, H, W]
                    # print(img_pos)
                    # fused_pos = img_pos
                    

                    if flow_pos is not None and img_pos is not None:
                        print(f"img_pos min={img_pos.min().item()}, max={img_pos.max().item()}, mean={img_pos.mean().item()}, std={img_pos.std().item()}")
                        print(f"flow_pos min={flow_pos.min().item()}, max={flow_pos.max().item()}, mean={flow_pos.mean().item()}, std={flow_pos.std().item()}")
                        fused_pos = (img_pos + flow_pos) / 2
                    else:
                        fused_pos = img_pos  # fallback: 只用 img_pos

                    all_positions.append(fused_pos)
                    # else:

                    #     print(f"camid{cam_id}")
                    #     print(f"flow_feat: {flow_feat}")
                    #     print(f"flow_pos: {flow_pos}")
                    #     raise ValueError("Both img_pos and flow_pos should be non-None")
                else:
                    all_features.append(img_feat)
                    all_positions.append(img_pos)
                    

            # 后续拼接操作
            src = torch.cat(all_features, dim=3)  # [bs, hidden_dim, H, W*num_cam]
            pos = torch.cat(all_positions, dim=3)  # [bs, hidden_dim, H, W*num_cam]
            # src = torch.cat(all_features, dim=3)  
            # pos = torch.cat(all_positions, dim=3)  
            
           
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
```
2. 加入了独立的光流提取器，使用raft_small模型
```
class AttentionFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)  # 改：不要强压到 hidden_dim

    def forward(self, x):
        attn = self.attention(x)
        out = x * attn
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
        
```
3. 另外导入了flow_backbone，用于提取光流特征
```import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBackbone(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=256):
        super().__init__()
        # 假设输入是光流图像 (B, 2, H, W)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # 位置编码相关（如果需要）
        self.num_channels = hidden_dim  # 用于后续投影层的输入通道数
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return [x], [None]  # 返回特征列表和位置编码列表（兼容原始接口）
```

## transformer
进行了一些数据对其上的debug
```
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self.src_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        bs = src.shape[0]

        if len(src.shape) == 4:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if c != self.d_model:
                src = self.src_proj(src)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        else:
            bs, seq_len, c = src.shape
            if c != self.d_model:
                src = self.src_proj(src)
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).expand(-1, bs, -1)

        query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)
        tgt = torch.zeros_like(query_embed)

        # Ensure latent/proprio are (batch, 512)
        latent_input = latent_input.squeeze(1) if latent_input.dim() == 3 else latent_input
        proprio_input = proprio_input.squeeze(1) if proprio_input.dim() == 3 else proprio_input
        # latent_input, proprio_input: [batch, dim] → ensure [batch, dim]
        if latent_input.dim() == 1:
            latent_input = latent_input.unsqueeze(0)  # [1, dim]
        if proprio_input.dim() == 1:
            proprio_input = proprio_input.unsqueeze(0)

        if latent_input.shape[-1] != self.d_model:
            latent_input = self.src_proj(latent_input)
        if proprio_input.shape[-1] != self.d_model:
            proprio_input = self.src_proj(proprio_input)

        # # Fix batch dim mismatch
        # if proprio_input.shape[0] != latent_input.shape[0]:
        #     if proprio_input.shape[0] == 1:
        #         proprio_input = proprio_input.expand(latent_input.shape[0], -1)
        #     else:
        #         proprio_input = proprio_input[:latent_input.shape[0]]
        # ensure batch match
        batch_size = src.shape[1]  # from image (after flatten and permute)
        if latent_input.shape[0] != batch_size:
            if latent_input.shape[0] == 1:
                latent_input = latent_input.expand(batch_size, -1)
            else:
                latent_input = latent_input[:batch_size]
        if proprio_input.shape[0] != batch_size:
            if proprio_input.shape[0] == 1:
                proprio_input = proprio_input.expand(batch_size, -1)
            else:
                proprio_input = proprio_input[:batch_size]
        addition_input = torch.stack([latent_input, proprio_input], axis=0)

        src = torch.cat([addition_input, src], axis=0)

        if additional_pos_embed.dim() == 2:
            additional_pos_embed = additional_pos_embed.unsqueeze(1)
        if additional_pos_embed.shape[1] != pos_embed.shape[1]:
            additional_pos_embed = additional_pos_embed.expand(-1, pos_embed.shape[1], -1)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # 🔥 Call your custom decoder (not PyTorch default)
        hs = self.decoder(
            tgt, memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed
        )
        hs = hs.transpose(1, 2)
        return hs
 ```
        


## **特征融合**
1. prev_image和cur_image通过raft计算光流并归一化
```flow = self.flow_extractor(prev_image, cur_image)      # [bs, num_cam, 2, H, W]
flow = torch.clamp(flow, -max_flow, max_flow) / max_flow
```
2. 过backbone泛化成feature和position
3. 过alpha conv和flow融合
``` 
self.alpha_conv = nn.Sequential(
                nn.Conv2d(2 * 512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Sigmoid()
            )
        self.input_proj_flow = n
```
4. 过transformer
```
hs = self.transformer(
                src=src,
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pos,
                latent_input=latent_input,
                proprio_input=proprio_input,
                additional_pos_embed=self.additional_pos_embed.weight
            )[0]
```