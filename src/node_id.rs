#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(u32);

impl NodeId {
    pub(crate) fn new(id: usize) -> Self {
        Self(u32::try_from(id).unwrap())
    }

    pub(crate) fn i(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}
