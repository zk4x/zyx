#[derive(Debug, Clone, Copy)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    //CF32,
    //CF64,
    U8,
    I8,
    I16,
    I32,
    I64,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
        })
    }
}
