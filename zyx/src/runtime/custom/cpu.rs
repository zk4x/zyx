use crate::dtype::DType;
use crate::runtime::custom::Custom;
use crate::scalar::Scalar;
use alloc::vec::Vec;
use alloc::string::String;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

#[derive(Debug)]
pub struct CPUError {
    info: String,
}

pub(crate) struct CPURuntime {}

pub(crate) enum CPUBuffer {
    #[cfg(feature = "half")]
    BF16(Vec<bf16>),
    #[cfg(feature = "half")]
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    #[cfg(feature = "complex")]
    CF32(Vec<Complex<f32>>),
    #[cfg(feature = "complex")]
    CF64(Vec<Complex<f64>>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl CPUBuffer {
    fn len(&self) -> usize {
        return match self {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(x) => x.len(),
            #[cfg(feature = "half")]
            CPUBuffer::F16(x) => x.len(),
            CPUBuffer::F32(x) => x.len(),
            CPUBuffer::F64(x) => x.len(),
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(x) => x.len(),
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(x) => x.len(),
            CPUBuffer::U8(x) => x.len(),
            CPUBuffer::I8(x) => x.len(),
            CPUBuffer::I16(x) => x.len(),
            CPUBuffer::I32(x) => x.len(),
            CPUBuffer::I64(x) => x.len(),
            CPUBuffer::Bool(x) => x.len(),
        };
    }

    fn dtype(&self) -> DType {
        return match self {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(_) => DType::BF16,
            #[cfg(feature = "half")]
            CPUBuffer::F16(_) => DType::F16,
            CPUBuffer::F32(_) => DType::F32,
            CPUBuffer::F64(_) => DType::F64,
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(_) => DType::CF32,
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(_) => DType::CF64,
            CPUBuffer::U8(_) => DType::U8,
            CPUBuffer::I8(_) => DType::I8,
            CPUBuffer::I16(_) => DType::I16,
            CPUBuffer::I32(_) => DType::I32,
            CPUBuffer::I64(_) => DType::I64,
            CPUBuffer::Bool(_) => DType::Bool,
        };
    }
}

impl Custom for CPURuntime {
    type Buffer = CPUBuffer;
    type Error = CPUError;
    fn initialize() -> Result<Self, CPUError> {
        Ok(Self {})
    }

    fn store_mem<T: Scalar>(&self, data: Vec<T>) -> Result<Self::Buffer, CPUError> {
        Ok(match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => CPUBuffer::BF16(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "half")]
            DType::F16 => CPUBuffer::F16(unsafe { core::mem::transmute(data) }),
            DType::F32 => CPUBuffer::F32(unsafe { core::mem::transmute(data) }),
            DType::F64 => CPUBuffer::F64(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "complex")]
            DType::CF32 => CPUBuffer::CF32(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "complex")]
            DType::CF64 => CPUBuffer::CF64(unsafe { core::mem::transmute(data) }),
            DType::U8 => CPUBuffer::U8(unsafe { core::mem::transmute(data) }),
            DType::I8 => CPUBuffer::I8(unsafe { core::mem::transmute(data) }),
            DType::I16 => CPUBuffer::I16(unsafe { core::mem::transmute(data) }),
            DType::I32 => CPUBuffer::I32(unsafe { core::mem::transmute(data) }),
            DType::I64 => CPUBuffer::I64(unsafe { core::mem::transmute(data) }),
            DType::Bool => CPUBuffer::Bool(unsafe { core::mem::transmute(data) }),
        })
    }

    fn load_mem<T: Scalar>(
        &self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CPUError> {
        assert_eq!(buffer.len(), length);
        if T::dtype() != buffer.dtype() {
            return Err(CPUError { info: "Wrong buffer dtype".into() });
        }
        let data: &Vec<T> = match buffer {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "half")]
            CPUBuffer::F16(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::F32(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::F64(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::U8(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I8(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I16(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I32(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I64(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::Bool(data) => unsafe { core::mem::transmute(data) },
        };
        return Ok(data.clone());
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CPUError> {
        // or nothing at all, but lets be explicit
        core::mem::drop(buffer);
        Ok(())
    }
}

/*
#define min(x, y) ((x) < (y) ? (x) : (y))

#if MR == 16 && NR == 6
#define kernel kernel_16x6

#elif MR == 8 && NR == 12
#define kernel kernel_8x12

#elif MR == 8 && NR == 13
#define kernel kernel_8x13

#elif MR == 8 && NR == 14
#define kernel kernel_8x14

#elif MR == 8 && NR == 8
#define kernel kernel_8x8

#else
#define kernel kernel_16x6

#endif

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

void pack_panelB(float* B, float* blockB_packed, const int nr, const int kc, const int K) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, const int nc, const int kc, const int K) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for (int j = 0; j < nc; j += NR) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

void pack_panelA(float* A, float* blockA_packed, const int mr, const int kc, const int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, const int mc, const int kc, const int M) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for (int i = 0; i < mc; i += MR) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void matmul(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int nr = min(NR, nc - jr);
                        const int mr = min(MR, mc - ir);
                        kernel(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                               &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
            }
        }
    }
}
*/
