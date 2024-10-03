// nanoGPT, credit goes to Andrej Karpathy and great minds who invented parts of this model
// https://github.com/karpathy/nanoGPT

use zyx::{Tensor, ZyxError, DType};
use zyx_nn::{Module, Linear, LayerNorm, CausalSelfAttention};

#[derive(Module)]
struct GPTConfig {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    dropout: f32,
    bias: bool,
    dtype: DType,
}

impl Default for GPTConfig {
    fn default() -> Self {
        return GPTConfig {
            block_size: 1024,
            vocab_size: 50304, // GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.0,
            bias: true, // True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
           dtype: DType::F32,
        }
    }
}

#[derive(Module)]
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout: f32,
}

impl MLP {
    fn new(config: GPTConfig) -> Result<MLP, ZyxError> {
        Ok(MLP {
            c_fc: Linear::new(config.n_embd, 4*config.n_embd, config.bias, config.dtype)?,
            c_proj: Linear::new(4*config.n_embd, config.n_embd, config.bias, config.dtype)?,
            dropout: config.dropout,
        })
    }

    fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let mut x = self.c_fc.forward(x)?;
        x = x.gelu();
        x = self.c_proj.forward(x)?;
        x = x.dropout(self.dropout)?;
        return Ok(x);
    }
}

#[derive(Module)]
struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(config: GPTConfig) -> Result<Block, ZyxError> {
        Ok(Block {
            ln_1: LayerNorm::new(config.n_embd, config.bias, config.dtype)?,
            attn: CausalSelfAttention::new(config.n_embd, config.n_head, config.bias, config.dropout, config.dtype)?,
            ln_2: LayerNorm::new(config.n_embd, config.bias, config.dtype)?,
            mlp: MLP::new(config)?,
        })
    }

    fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let mut x = x.into();
        x = &x + self.attn.forward(self.ln_1.forward(&x)?)?;
        x = &x + self.mlp.forward(self.ln_2.forward(&x)?)?;
        return Ok(x);
    }
}

#[derive(Module)]
struct GPT {
    config: GPTConfig,
}

impl GPT {
    fn new(config: GPTConfig) -> Result<GPT, ZyxError> {
        assert!(config.vocab_size > 0);
        assert!(config.block_size > 0);

        let gpt = GPT {
            config,
            wte: Embedding::new(config.vocab_size, config.n_embd)?,
            wpe: Embedding::new(config.block_size, config.n_embd)?,
            h: (0..config.n_layer).map(|_| Block::new(config).unwrap()),
            ln_f: LayerNorm::new(config.n_embd, config.bias, config.dtype)?,
            lm_head: Linear::new(config.n_embd, config.vocab_size, config.bias, config.dtype)?,
        };

        Ok(gpt)
    }
}

/*
def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
*/

fn main() {}
