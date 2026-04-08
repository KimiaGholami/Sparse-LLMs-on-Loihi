from transformers import PretrainedConfig

class GatedDeltaProductConfig(PretrainedConfig):
    model_type = "gated_deltaproduct"  # unique model_type key

    def __init__(
        self,
        # Architecture
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_hidden_layers: int = 21,
        # GatedDeltaProduct-specific
        num_heads: int = 6,
        head_dim: int = 256,
        expand_v: float = 2.0,
        num_householder: int = 2,        # <-- second-order: 2 Householder transforms
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_forget_gate: bool = True,
        allow_neg_eigval: bool = True,
        attn_mode: str = "chunk",
        norm_eps: float = 1e-5,
        # FFN
        hidden_ratio: float = 4.0,
        hidden_act: str = "swish",
        intermediate_size: int = None,
        # Common
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.006,
        fuse_cross_entropy: bool = True,
        fuse_norm: bool = True,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.num_householder = num_householder
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.attn_mode = attn_mode
        self.norm_eps = norm_eps
        self.hidden_ratio = hidden_ratio
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )