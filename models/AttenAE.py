import torch
import torch.nn as nn


class AttenAEFusionModel(nn.Module):
    def __init__(self, config):
        super(AttenAEFusionModel, self).__init__()

        self.dropout_rate = config["dropout_rate"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.span_input_dim = config["span_input_dim"]
        self.log_input_dim = config["log_input_dim"]
        # self.span_max_length = config["span_max_length"]
        # self.log_max_length = config["log_max_length"]

        # Encoders for adjusting dimensions of span and log data
        self.span_encoder = nn.Sequential(
            nn.Linear(self.span_input_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        self.log_encoder = nn.Sequential(
            nn.Linear(self.log_input_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

        # MultiheadAttention for fusion
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                                          batch_first=True)

        # Decoders for reconstructing span and log data from the fused representation
        self.span_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.span_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        self.log_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.log_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, span_data, log_data):
        # Attention to fuse span and log data
        spans_mask = self.create_padding_mask(span_data, self.num_heads)

        span_encoded = self.span_encoder(span_data)

        fusion_output, _ = self.multi_head_attention(query=span_encoded, key=log_data, value=log_data,
                                                     attn_mask=spans_mask)

        recon_span = self.span_decoder(fusion_output)
        recon_log = self.log_decoder(fusion_output)

        return recon_span, recon_log, fusion_output

    def create_padding_mask(self, seq, num_heads):
        mask = torch.eq(seq.sum(dim=-1), 0)
        mask = mask.unsqueeze(1).expand(-1, seq.size(1), -1)
        mask = mask.repeat(num_heads, 1, 1)
        return mask



