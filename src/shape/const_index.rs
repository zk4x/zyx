use super::Shape;

pub trait ConstIndex<const IDX: i32>: Shape {
    fn const_at(&self) -> Self::D;
    fn const_mut_at(&mut self) -> &mut Self::D;
}

impl ConstIndex<0> for usize {
    fn const_at(&self) -> Self::D {
        *self
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        self
    }
}

impl ConstIndex<-1> for usize {
    fn const_at(&self) -> Self::D {
        *self
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        self
    }
}

impl ConstIndex<1> for (usize, usize) {
    fn const_at(&self) -> Self::D {
        self.1
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.1
    }
}

impl ConstIndex<0> for (usize, usize) {
    fn const_at(&self) -> Self::D {
        self.0
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.0
    }
}

impl ConstIndex<-1> for (usize, usize) {
    fn const_at(&self) -> Self::D {
        self.1
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.1
    }
}

impl ConstIndex<-2> for (usize, usize) {
    fn const_at(&self) -> Self::D {
        self.0
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.0
    }
}

impl ConstIndex<0> for i32 {
    fn const_at(&self) -> Self::D {
        *self
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self
    }
}

impl ConstIndex<-1> for i32 {
    fn const_at(&self) -> Self::D {
        *self
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self
    }
}

impl ConstIndex<1> for (i32, i32) {
    fn const_at(&self) -> Self::D {
        self.1
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.1
    }
}

impl ConstIndex<0> for (i32, i32) {
    fn const_at(&self) -> Self::D {
        self.0
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.0
    }
}

impl ConstIndex<-1> for (i32, i32) {
    fn const_at(&self) -> Self::D {
        self.1
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.1
    }
}

impl ConstIndex<-2> for (i32, i32) {
    fn const_at(&self) -> Self::D {
        self.0
    }

    fn const_mut_at(&mut self) -> &mut Self::D {
        &mut self.0
    }
}
