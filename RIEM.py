import torch
import torch.nn as nn

class CAN(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_size, dropout=0.1):
        super(CAN, self).__init__()
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.input_proj_tf = nn.LazyLinear(hidden_dim)
        self.input_proj_target = nn.LazyLinear(hidden_dim)

        self.query_tf = nn.LazyLinear(hidden_dim, bias=False)
        self.key_tf = nn.LazyLinear(hidden_dim, bias=False)
        self.value_tf = nn.LazyLinear(hidden_dim, bias=False)

        self.query_target = nn.LazyLinear(hidden_dim, bias=False)
        self.key_target = nn.LazyLinear(hidden_dim, bias=False)
        self.value_target = nn.LazyLinear(hidden_dim, bias=False)

        self.norm_fg = nn.LayerNorm(hidden_dim)
        self.norm_gf = nn.LayerNorm(hidden_dim)

        self.final_norm = nn.LayerNorm(hidden_dim * 2)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped
    
    def forward(self, tf, target):
        tf_proj = self.input_proj_tf(tf)  # [batch, feat_dim] -> [batch, hidden]
        target_proj = self.input_proj_target(target)        # [batch, feat_dim] -> [batch, hidden]
        
        tf = tf_proj.unsqueeze(1).float().to(tf_proj.device)  # [batch, 1, hidden]
        target = target_proj.unsqueeze(1).float().to(target_proj.device)           # [batch, 1, hidden]
        
        tf_orig, target_orig = tf, target

        mask_tf = torch.ones(tf.size()[:-1], dtype=torch.bool, device=tf.device)
        mask_target = torch.ones(target.size()[:-1], dtype=torch.bool, device=target.device)

        q_tf = self.apply_heads(self.query_tf(tf), self.num_heads, self.head_size) # [B, 1, 384] -> [B, 1, 8, 48]
        k_tf = self.apply_heads(self.key_tf(tf), self.num_heads, self.head_size)
        v_tf = self.apply_heads(self.value_tf(tf), self.num_heads, self.head_size)
    
        q_target = self.apply_heads(self.query_target(target), self.num_heads, self.head_size)
        k_target = self.apply_heads(self.key_target(target), self.num_heads, self.head_size)
        v_target = self.apply_heads(self.value_target(target), self.num_heads, self.head_size)

        logits_fg = torch.einsum('blhd, bkhd->blkh', q_tf, k_target) / (self.head_size ** 0.5) # [batch_size, 1, 8, 48] -> [batch_size, 1, 1, 8]
        logits_gf = torch.einsum('blhd, bkhd->blkh', q_target, k_tf) / (self.head_size ** 0.5)

        alpha_fg = self.alpha_logits(logits_fg, mask_tf, mask_target) # softmax
        alpha_gf = self.alpha_logits(logits_gf, mask_target, mask_tf)

        fg_out = torch.einsum('blkh, bkhd->blhd', alpha_fg, v_target).flatten(-2)
        fg_out = self.norm_fg(target_orig + self.dropout(fg_out))
        
        gf_out = torch.einsum('blkh, bkhd->blhd', alpha_gf, v_tf).flatten(-2)
        gf_out = self.norm_gf(tf_orig + self.dropout(gf_out))

        tf_embedding = fg_out
        target_embedding = gf_out

        tf_embed = tf_embedding.mean(1)
        target_embed = target_embedding.mean(1)

        query_embed = self.final_norm(torch.cat([tf_embed, target_embed], dim=1))
        return query_embed
