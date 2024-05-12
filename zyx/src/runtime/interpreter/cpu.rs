use crate::runtime::interpreter::{Interpreter, InterpreterError};

pub(crate) struct CPU {}

impl Interpreter for CPU {
    fn initialize() -> Result<Self, InterpreterError> {
        todo!()
    }
}