use crate::{dtype::Constant, runtime::{ir::{IRDType, IRKernel, IROp, Scope, Var}, node::{BOp, UOp}}};

pub(crate) enum WGSLError {}

pub(crate) struct WGSLDevice {}

pub(crate) struct WGSLProgram {}

impl WGSLDevice {
    fn compile(&mut self, kernel: &IRKernel, debug_asm: bool) -> Result<WGSLProgram, WGSLError> {
        let mut source = String::new();
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        /*
        @group(0) @binding(0) var<storage, read> g0: array<f32>;
        @group(0) @binding(1) var<storage, read> g1: array<f32>;
        @group(0) @binding(2) var<storage, write> g2: array<f32>;
        */

        for op in &kernel.ops[..6] {
            if let &IROp::Loop { id, len } = op {
                if id % 2 == 0 {
                    global_work_size[id as usize / 2] = len;
                } else {
                    local_work_size[id as usize / 2] = len;
                }
            } else {
                panic!()
            }
        }

        // Declare global variables
        for (id, (_, dtype, read_only)) in kernel.addressables.iter().enumerate() {
            source += &format!(
                "@group(0) @binding({id}) var<storage, {}> g{id}: array<{}>;\n",
                if *read_only { "read" } else { "write" },
                dtype.wgsl(),
            );
        }

        // NOTE Just gonna assume wgsl workgroup size is local work size
        source += &format!("@compute @workgroup_size({}, {}, {})", local_work_size[0], local_work_size[1], local_work_size[2]);
        source += &format!("fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n");

        // Declare register variables
        for (id, (dtype, read_only)) in kernel.registers.iter().enumerate() {
            source += &format!(
                "{indent}{} r{id}: {};\n",
                if *read_only { "let" } else { "var" },
                dtype.wgsl()
            );
        }

        // Add indices for global and local loops
        source += &format!(
            "  r0 = gid.x;   /* 0..{} */\n",
            global_work_size[0]
        );
        source += &format!(
            "  r1 = lid.x;   /* 0..{} */\n",
            local_work_size[0]
        );
        source += &format!(
            "  r2 = gid.y;   /* 0..{} */\n",
            global_work_size[1]
        );
        source += &format!(
            "  r3 = lid.y;   /* 0..{} */\n",
            local_work_size[1]
        );
        source += &format!(
            "  r4 = gid.z;   /* 0..{} */\n",
            global_work_size[2]
        );
        source += &format!(
            "  r5 = lid.z;   /* 0..{} */\n",
            local_work_size[2]
        );

        for op in kernel.ops[6..kernel.ops.len()-6].iter().copied() {
            match op {
                IROp::Set { z, len: _, value } => {
                    source += &format!("{indent}r{z} = {value};\n");
                }
                IROp::Load { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{} = {}[{}];\n", z.wgsl(), x.wgsl(), at.wgsl());
                }
                IROp::Store { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{}[{}] = {};\n", z.wgsl(), at.wgsl(), x.wgsl());
                }
                IROp::Unary { z, x, uop, dtype } => {
                    source += &match uop {
                        UOp::Cast(_) => format!("{indent}{} = ({}){};\n", z.wgsl(), dtype.wgsl(), x.wgsl()),
                        UOp::ReLU => format!("{indent}{} = max({}, 0);\n", z.wgsl(), x.wgsl()),
                        UOp::Neg => format!("{indent}{} = -{};\n", z.wgsl(), x.wgsl()),
                        UOp::Exp2 => format!("{indent}{} = exp2({});\n", z.wgsl(), x.wgsl()),
                        UOp::Log2 => format!("{indent}{} = log2({});\n", z.wgsl(), x.wgsl()),
                        UOp::Inv => format!("{indent}{} = 1/{};\n", z.wgsl(), x.wgsl()),
                        UOp::Sqrt => format!("{indent}{} = sqrt({});\n", z.wgsl(), x.wgsl()),
                        UOp::Sin => format!("{indent}{} = sin({});\n", z.wgsl(), x.wgsl()),
                        UOp::Cos => format!("{indent}{} = cos({});\n", z.wgsl(), x.wgsl()),
                        UOp::Not => format!("{indent}{} = !{};\n", z.wgsl(), x.wgsl()),
                        UOp::Nonzero => format!("{indent}{} = {} != 0;\n", z.wgsl(), x.wgsl()),
                    };
                }
                IROp::Binary {
                    z,
                    x,
                    y,
                    bop,
                    dtype: _,
                } => {
                    source += &format!(
                        "{indent}{} = {};\n",
                        z.wgsl(),
                        match bop {
                            BOp::Add => format!("{} + {}", x.wgsl(), y.wgsl()),
                            BOp::Sub => format!("{} - {}", x.wgsl(), y.wgsl()),
                            BOp::Mul => format!("{} * {}", x.wgsl(), y.wgsl()),
                            BOp::Div => format!("{} / {}", x.wgsl(), y.wgsl()),
                            BOp::Pow => format!("pow({}, {})", x.wgsl(), y.wgsl()),
                            BOp::Cmplt => format!("{} < {}", x.wgsl(), y.wgsl()),
                            BOp::Cmpgt => format!("{} > {}", x.wgsl(), y.wgsl()),
                            BOp::Max => format!("max({}, {})", x.wgsl(), y.wgsl()),
                            BOp::Or => format!("{} || {}", x.wgsl(), y.wgsl()),
                        }
                    );
                }
                IROp::MAdd {
                    z,
                    a,
                    b,
                    c,
                    dtype: _,
                } => {
                    source += &format!("{indent}{} = {} * {} + {};\n", z.wgsl(), a.wgsl(), b.wgsl(), c.wgsl());
                }
                IROp::Loop { id, len } => {
                    source += &format!(
                        "{indent}for (var r{id}: u32 = 0; r{id} < {len}; r{id} = r{id} + 1) {{\n"
                    );
                    indent += "  ";
                }
                IROp::EndLoop { .. } => {
                    indent.pop();
                    indent.pop();
                    source += &format!("{indent}}}\n");
                }
                IROp::Barrier { scope } => {
                    source += &format!(
                        "{indent}barrier(CLK_{}AL_MEM_FENCE);\n",
                        match scope {
                            Scope::Global => "GLOB",
                            Scope::Local => "LOC",
                            Scope::Register => panic!(),
                        }
                    );
                }
            }
        }
        source += "}\n";

        let mut global_work_size = global_work_size;
        let local_work_size = local_work_size;
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        for (i, lwd) in local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
        let mut pragma = format!("");
        if source.contains("double") {
            pragma += &"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = format!("{pragma}__kernel void {name}{source}");
        if debug_asm {
            println!("{source}");
        }

        todo!()
    }
}

impl IRDType {
    fn wgsl(&self) -> &str {
        return match self {
            #[cfg(feature = "half")]
            IRDType::BF16 => todo!("WIP"),
            #[cfg(feature = "half")]
            IRDType::F16 => "f16",
            IRDType::F32 => "f32",
            IRDType::F64 => "f64",
            #[cfg(feature = "complex")]
            IRDType::CF32 => todo!("WIP"),
            #[cfg(feature = "complex")]
            IRDType::CF64 => todo!("WIP"),
            IRDType::U8 => "u8",
            IRDType::I8 => "i8",
            IRDType::I16 => "i16",
            IRDType::I32 => "int",
            IRDType::I64 => "i64",
            IRDType::Bool => "bool",
            IRDType::U32 => "u32",
        };
    }
}

impl Constant {
    fn wgsl(&self) -> String {
        use core::mem::transmute as t;
        match self {
            #[cfg(feature = "half")]
            Constant::F16(x) => format!("{}f", unsafe { t::<_, half::f16>(*x) }),
            #[cfg(feature = "half")]
            Constant::BF16(x) => format!("{}f", unsafe { t::<_, half::bf16>(*x) }),
            Constant::F32(x) => format!("{}f", unsafe { t::<_, f32>(*x) }),
            Constant::F64(x) => format!("{}f", unsafe { t::<_, f64>(*x) }),
            #[cfg(feature = "complex")]
            Constant::CF32(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            #[cfg(feature = "complex")]
            Constant::CF64(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            Constant::U8(x) => format!("{x}"),
            Constant::I8(x) => format!("{x}"),
            Constant::I16(x) => format!("{x}"),
            Constant::U32(x) => format!("{x}"),
            Constant::I32(x) => format!("{x}"),
            Constant::I64(x) => format!("{x}"),
            Constant::Bool(x) => format!("{x}"),
        }
    }
}

impl Var {
    fn wgsl(&self) -> String {
        match self {
            Var::Id(id, scope) => format!("{scope}{id}"),
            Var::Const(value) => format!("{}", value.wgsl()),
        }
    }
}
