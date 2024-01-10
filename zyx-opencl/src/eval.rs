use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use zyx_core::node::Node;
use zyx_core::tensor::Id;
use crate::inner::Buffer;

/// This function evaluates concrete buffer that we know can be directly evaluated,
/// that is it all of it's leafs are already evaluated.
pub(crate) fn evaluate_buffer(buffers: &BTreeMap<Id, Buffer>, order: &[Id], nodes: &[Node], x: Id) {
    // create ordered list of nodes that need to be evaluated
    let mut temp = alloc::vec![x];
    let mut porder = Vec::new();
    while let Some(nid) = temp.pop() {
        if buffers.contains_key(&nid) {
            continue
        }
        porder.extend(nodes[nid.i()].parameters())
    }
    porder.sort_by_cached_key(|nid| order[nid.i()]);
    // TODO perhaps we can cache it somewhere here using nodes,
    // otherwise when using strings to cache, it can take
    // up to a few MB of RAM

    /*
    // We can use Kernel struct as cache :)
    Perhaps this is too much complexity just yet
    enum Scope {
        Global,
        Local,
        Private,
    }

    struct KId(usize, Scope);

    // These are all IDs into ops, with Leafs having IDs into parameters
    enum Op {
        Leaf(KId),
        CastF32(KId),
        Exp(KId),
        Tanh(KId),
        Add(KId, KId),
    }

    struct Kernel {
        global_work_size: [usize; 2],
        local_work_size: [usize; 2],
        register_work_size: usize,
        parameters: Vec<(*mut c_void, View, DType)>,
        tiles: Vec<(usize, DType)>,
        accumulators: Vec<(usize, DType)>,
        before_reduce_ops: Vec<Op>,
        after_reduce_ops: Vec<Op>,
    }*/

    let mut kernel_parameters = Vec::new();
    for nid in porder {
        if let Some(x) = buffers.get(&nid) {
            kernel_parameters.push(x.mem);
        } else {
            todo!()
        }
    }
}
