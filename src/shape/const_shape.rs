/*
Operations needed:
reduce, reshape, expand, permute
 */

extern crate alloc;

trait Shape {
    const RANK: usize;
    //type Strides: Shape;
    fn strides() -> alloc::vec::Vec<usize>;
    fn numel() -> usize;
}

struct Sh0 {}

struct Sh1<const D0: usize> {}

struct Sh2<const D0: usize, const D1: usize> {}

struct Sh3<const D0: usize, const D1: usize, const D2: usize> {}

/*impl<const D0: usize, const D1: usize, const D2: usize> Shape for Sh3<D0, D1, D2> {
    const RANK: usize = 3;
    //type Strides = Sh3<{ D1*D2 }, D2, 1>;
    fn strides() -> alloc::vec::Vec<usize> {
        alloc::vec![D1*D2, D2, 1]
    }
}*/

struct Sh4<const D0: usize, const D1: usize, const D2: usize, const D3: usize> {}

struct Sh5<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> {}




trait Axes {
    const RANK: usize;
    fn strides() -> alloc::vec::Vec<usize>;
    fn argsort() -> alloc::vec::Vec<usize>;
}

struct AxA {}

struct Ax1<const D0: i32> {}

struct Ax2<const D0: i32, const D1: i32> {}

struct Ax3<const D0: i32, const D1: i32, const D2: i32> {}

struct Ax4<const D0: i32, const D1: i32, const D2: i32, const D3: i32> {}

struct Ax5<const D0: i32, const D1: i32, const D2: i32, const D3: i32, const D4: i32> {}
