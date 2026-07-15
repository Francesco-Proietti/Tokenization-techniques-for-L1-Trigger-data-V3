import torch
import torch.nn as nn

import lightning as pl


# GPT block---------------------------
class GPTBlock(nn.Module):
    """
    Decoder-only Transformer block (Pre-LayerNorm).

    Structure:
        x = x + SelfAttention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)

        hidden_dim = mlp_ratio * d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
    ):
        """
        Parameters
        ----------
        x : (B, L, D)

        attn_mask :
            causal mask (L,L)

        key_padding_mask :
            (B,L)
            True = PAD
            False = token valido
        """

        # -----------------------
        # Self Attention
        # -----------------------

        h = self.ln1(x)

        h, _ = self.attn(
            query=h,
            key=h,
            value=h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = x + self.dropout1(h)

        # -----------------------
        # Feed Forward
        # -----------------------

        h = self.ln2(x)

        h = self.mlp(h)

        x = x + h

        return x


#GPT backbone--------------------------
class GPTBackbone(nn.Module):
    """
    Decoder-only Transformer backbone.

    Output:
        hidden states (B, L, D)

    no final head
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # -----------------------------
        # Embeddings
        # -----------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # Transformer blocks
        # -----------------------------

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # -----------------------------

        self.norm = nn.LayerNorm(d_model)

    def _build_causal_mask(
        self,
        seq_len,
        device,
    ):

        return torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):
        """
        Parameters
        ----------

        tokens:
            (B,L)

        attention_mask:
            (B,L)
        """

        B, L = tokens.shape

        device = tokens.device

        # -----------------------------------

        positions = torch.arange(
            L,
            device=device,
        ).unsqueeze(0)

        x = (
            self.token_embedding(tokens)
            +
            self.position_embedding(positions)
        )

        x = self.dropout(x)

        # -----------------------------------

        causal_mask = self._build_causal_mask(
            L,
            device,
        )

        if attention_mask is None:

            key_padding_mask = None

        else:

            key_padding_mask = ~attention_mask

        # -----------------------------------

        for block in self.blocks:

            x = block(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        x = self.norm(x)

        return x


#GPT for pretraining--------------------------
class GPTForPretraining(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=256,
        n_layers=4,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = GPTBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying (GPT-2)
        self.lm_head.weight = self.backbone.token_embedding.weight

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):

        hidden = self.backbone(
            tokens,
            attention_mask,
        )

        logits = self.lm_head(hidden)

        return logits

