// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Graph visualizer.
//!
//! Compiled only behind the `viz` feature (`--features viz`). Every realized
//! graph adds one tab to a web page served at `0.0.0.0:4242`. The tab shows
//! the ExecPlan as an interactive graph; clicking a kernel shows its
//! pre-linearize IR (`ZYX_DEBUG=4` view), optimized IR (`ZYX_DEBUG=8` view)
//! and generated source code (`ZYX_DEBUG=16` view).
//!
//! Kernels are captured during generation (right after compilation), because
//! optimized kernels are transient. The capture holds both the pre-linearize
//! kernel (`ZYX_DEBUG=4` view) and the autotune winner kernel, which is the
//! optimized IR (`ZYX_DEBUG=8` view); the generated code is derived lazily on
//! request from the winner.
mod page;
mod server;

use crate::{
    Map,
    backend::{DeviceInfo, ProgramId},
    graph::{ClassId, ExecPlan, Graph},
    kernel::Kernel,
};
use std::sync::{Arc, Mutex, OnceLock};

/// Base offset for kernel node ids in the plan visualization, so they never
/// collide with class node ids (which are raw `ClassId`s).
const LAUNCH_NODE_BASE: usize = 1 << 30;

/// A kernel captured during generation together with everything needed to
/// show the optimized IR (`ZYX_DEBUG=8` view) and the generated source code.
#[derive(Clone)]
pub(crate) struct KernelCapture {
    /// Pre-linearize kernel (`ZYX_DEBUG=4` view).
    pub(crate) sched_kernel: Kernel,
    /// Winner kernel of the autotune search for this program.
    pub(crate) winner: Kernel,
    /// Device info of the device this program was compiled for.
    pub(crate) dev_info: DeviceInfo,
    /// Human readable device kind ("CUDA", "OpenCL", ...).
    pub(crate) device_label: &'static str,
    /// CUDA compute capability (for PTX codegen).
    pub(crate) cc: Option<[i32; 2]>,
    /// Whether the C backend was compiled with OpenMP support.
    pub(crate) has_openmp: bool,
}

/// One node of the plan visualization.
pub(crate) struct PlanNode {
    id: usize,
    label: String,
    /// Index into `GraphViz::kernels`, or `-1` for non-kernel nodes.
    kernel: i64,
}

/// One realized graph: plan structure plus captured kernels.
pub(crate) struct GraphViz {
    name: String,
    nodes: Vec<PlanNode>,
    edges: Vec<(usize, usize, String)>,
    kernels: Vec<Option<KernelCapture>>,
}

/// Shared visualizer state.
pub(crate) struct VizData {
    graphs: Vec<GraphViz>,
    /// Staged captures keyed by program id, moved into graphs on snapshot.
    staged: Map<ProgramId, KernelCapture>,
}

/// Handle to the visualizer state. Stored as `Runtime::graph_viz`.
/// Cheap to construct (no allocation until first capture).
pub struct Viz {
    inner: OnceLock<Arc<Mutex<VizData>>>,
}

impl Viz {
    /// Create an inactive visualizer handle.
    pub const fn new() -> Self {
        Self { inner: OnceLock::new() }
    }

    fn data(&self) -> Arc<Mutex<VizData>> {
        self.inner
            .get_or_init(|| {
                let data = Arc::new(Mutex::new(VizData { graphs: Vec::new(), staged: Map::default() }));
                server::spawn(Arc::clone(&data));
                data
            })
            .clone()
    }

    /// Stage a kernel captured right after it was compiled into `program`.
    pub(crate) fn record(&self, program_id: ProgramId, cap: KernelCapture) {
        self.data().lock().unwrap().staged.insert(program_id, cap);
    }

    /// Add one tab for `plan` compiled from `graph`.
    pub(crate) fn snapshot(&self, graph: &Graph, plan: &ExecPlan) {
        let data = self.data();
        let mut d = data.lock().unwrap();
        let name = format!("graph {}", d.graphs.len());

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut seen_edges = crate::Set::default();
        let mut class_nodes: Map<ClassId, usize> = Map::default();
        let mut kernels = Vec::new();

        let class_node = |cid: ClassId, nodes: &mut Vec<PlanNode>, class_nodes: &mut Map<ClassId, usize>| -> usize {
            *class_nodes.entry(cid).or_insert_with(|| {
                let id = usize::from(cid);
                nodes.push(PlanNode { id, label: class_label(graph, cid), kernel: -1 });
                id
            })
        };

        for node in &plan.nodes {
            let crate::graph::plan::ExecNode::Launch { program_id, load_classes, store_classes } = node else {
                continue;
            };
            let kid = kernels.len();
            let (label, captured) = match d.staged.get(program_id) {
                Some(cap) => (format!("kernel {kid}\n{}", cap.device_label), Some(cap)),
                None => ("AOT\nkernel".to_string(), None),
            };
            kernels.push(captured.map(|cap| KernelCapture {
                sched_kernel: cap.sched_kernel.clone(),
                winner: cap.winner.clone(),
                dev_info: cap.dev_info.clone(),
                device_label: cap.device_label,
                cc: cap.cc,
                has_openmp: cap.has_openmp,
            }));
            let launch_id = LAUNCH_NODE_BASE + kid;
            nodes.push(PlanNode { id: launch_id, label, kernel: kid as i64 });
            for &c in &**load_classes {
                let from = class_node(c, &mut nodes, &mut class_nodes);
                if seen_edges.insert((from, launch_id)) {
                    edges.push((from, launch_id, "load".to_string()));
                }
            }
            for &c in &**store_classes {
                let to = class_node(c, &mut nodes, &mut class_nodes);
                if seen_edges.insert((launch_id, to)) {
                    edges.push((launch_id, to, "store".to_string()));
                }
            }
        }

        d.staged.clear();
        d.graphs.push(GraphViz { name, nodes, edges, kernels });
    }
}

/// Label for a class node: id, dtype and resolved shape dims.
fn class_label(graph: &Graph, cid: ClassId) -> String {
    let dtype = graph.dtype(cid);
    let shape: Vec<String> = graph.shape(cid).iter().map(|&d| dim_label(graph, d)).collect();
    format!("c{}\n{:?} {}", cid.0, dtype, shape.join("x"))
}

/// Resolve a dim class to its constant value, or "?" if dynamic.
fn dim_label(graph: &Graph, dim: ClassId) -> String {
    let Some(&nid) = graph.classes[dim].nodes.first() else {
        return "?".to_string();
    };
    match &graph.nodes[nid].node {
        crate::graph::Node::Const { value, .. } => match value.as_dim() {
            Some(v) => v.to_string(),
            None => "?".to_string(),
        },
        _ => "?".to_string(),
    }
}

/// Codegen target selectable in the web UI dropdown.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    CudaC,
    Ptx,
    OpenCL,
    C,
    Spirv,
}

impl Target {
    /// All targets in dropdown order.
    pub(crate) const ALL: [Self; 5] = [Self::CudaC, Self::Ptx, Self::OpenCL, Self::C, Self::Spirv];

    /// Query-parameter name of this target.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CudaC => "cuda",
            Self::Ptx => "ptx",
            Self::OpenCL => "opencl",
            Self::C => "c",
            Self::Spirv => "spirv",
        }
    }

    /// Parse a query-parameter name.
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "cuda" => Some(Self::CudaC),
            "ptx" => Some(Self::Ptx),
            "opencl" => Some(Self::OpenCL),
            "c" => Some(Self::C),
            "spirv" => Some(Self::Spirv),
            _ => None,
        }
    }
}

/// The optimized kernel (`ZYX_DEBUG=8` view): the winner kernel stored by the
/// autotune search when the program was compiled.
fn derive_optimized(cap: &KernelCapture) -> Kernel {
    cap.winner.clone()
}

/// Generate the source code for `target` from the derived optimized kernel.
/// Errors are rendered into the returned string (the UI shows them inline).
pub(crate) fn generate_source(cap: &KernelCapture, target: Target) -> String {
    let kernel = derive_optimized(cap);
    match target {
        Target::CudaC => match kernel.generate_cuda(&cap.dev_info, "zyx_viz") {
            Ok(source) => source,
            Err(e) => format!("generate_cuda failed: {e:?}"),
        },
        Target::Ptx => {
            let Some(cc) = cap.cc else {
                return "PTX requires a CUDA device (no compute capability recorded).".to_string();
            };
            match kernel.generate_ptx(cc, &cap.dev_info) {
                Ok((ptx, _, _)) => String::from_utf8_lossy(&ptx).into_owned(),
                Err(e) => format!("generate_ptx failed: {e:?}"),
            }
        }
        Target::OpenCL => match kernel.generate_opencl(&cap.dev_info, "zyx_viz") {
            Ok(source) => source,
            Err(e) => format!("generate_opencl failed: {e:?}"),
        },
        Target::C => match kernel.generate_c(&cap.dev_info, cap.has_openmp, "zyx_viz") {
            Ok(source) => source,
            Err(e) => format!("generate_c failed: {e:?}"),
        },
        Target::Spirv => match kernel.generate_spirv(false) {
            Ok(words) => crate::codegen::spirv::debug_string(&words),
            Err(e) => format!("generate_spirv failed: {e:?}"),
        },
    }
}
