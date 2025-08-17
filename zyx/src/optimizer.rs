pub struct Optimizer(Vec<Box<dyn Optimization>>);

trait Optimization {}

struct LocalWorkSize {}

impl Optimization for LocalWorkSize {}
