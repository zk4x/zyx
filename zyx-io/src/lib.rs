use std::io;
use std::path::Path;
use zyx::Tensor;

pub trait TensorLoad: Sized {
    fn load(path: impl AsRef<std::path::Path>) -> Result<Self, io::Error>;
}

pub trait TensorSave {
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), io::Error>;
}

/*impl TensorLoad for Tensor {
    fn load(path: impl AsRef<std::path::Path>) -> Result<Vec<Tensor>, io::Error> {
        todo!()
    }
}

impl TensorSave for Tensor {
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), io::Error> {
        todo!()
    }
}*/

// TODO this probably can't work, TensorLoad can probably be implemented only for From<Iterator<...>>
// impl<'a, M: From<Iterator<&'a Tensor>>> TensorLoad for M { ... }
impl<'a, M: IntoIterator<Item = &'a mut Tensor>> TensorLoad for M {
    fn load(path: impl AsRef<Path>) -> Result<M, io::Error> {
        todo!()
    }
}

impl<'a, M: IntoIterator<Item = &'a Tensor>> TensorSave for M {
    fn save(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        todo!()
    }
}
