use zyx::{DType, IntoShape, Tensor, ZyxError};
use zyx_derive::Module;

/// Applies a 2D convolution over an input signal composed of several input planes.
///
/// See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d
#[derive(Module)]
pub struct Conv2d {
    stride: Vec<usize>,
    dilation: Vec<usize>,
    groups: Vec<usize>,
    padding: Vec<usize>,
    /// weight
    pub weight: Tensor,
    /// bias
    pub bias: Option<Tensor>,
}

impl Conv2d {
    /// Initialize Conv2d
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: impl IntoShape,
        stride: impl IntoShape,
        padding: impl IntoShape,
        dilation: impl IntoShape,
        groups: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        let mut kernel_size: Vec<usize> = kernel_size.into_shape().collect();
        kernel_size.push(2);
        let scale = 1f32 / ((in_channels * kernel_size.iter().product::<usize>()) as f32).sqrt();
        let mut weight_shape = vec![out_channels, in_channels / groups];
        weight_shape.extend(kernel_size);
        Ok(Conv2d {
            stride: stride.into_shape().collect(),
            dilation: dilation.into_shape().collect(),
            groups: groups.into_shape().collect(),
            padding: padding.into_shape().collect(),
            weight: Tensor::uniform(weight_shape, -scale..scale)?.cast(dtype),
            bias: if bias {
                Some(Tensor::uniform(out_channels, -scale..scale)?.cast(dtype))
            } else {
                None
            },
        })
    }

    /// Forward conv2d layer
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        x.into().conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)
    }
}

/*def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0,
              dilation=1, groups=1, bias=True):
  self.kernel_size = make_tuple(kernel_size, 2)
  if isinstance(padding, str):
    if padding.lower() != 'same': raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
    if stride != 1: raise ValueError("padding='same' is not supported for strided convolutions")
    pad = [(d*(k-1)//2, d*(k-1) - d*(k-1)//2) for d,k in zip(make_tuple(dilation, len(self.kernel_size)), self.kernel_size[::-1])]
    padding = tuple(flatten(pad))
  self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
  scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
  self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
  self.bias: Tensor|None = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

def __call__(self, x:Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)*/
