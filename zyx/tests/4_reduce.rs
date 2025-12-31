use zyx::{Tensor, ZyxError};

#[test]
fn sum_1() -> Result<(), ZyxError> {
    {
        let x = Tensor::from([2, 4]);
        assert_eq!(x.sum_all(), 6);
    }
    Ok(())
}

#[test]
fn sum_2() -> Result<(), ZyxError> {
    {
        let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
        let x0 = x.sum([-1])?;
        let x1 = x.sum([-2])?;
        let x2 = x.sum_all();
        assert_eq!(x0, [8, 10, 18]);
        assert_eq!(x1, [15, 8, 13]);
        assert_eq!(x2, [36]);
    }
    Ok(())
}

#[test]
fn sum_3() -> Result<(), ZyxError> {
    {
        let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
        assert_eq!(x.sum([0])?, [3, 9, 4]);
        assert_eq!(x.sum([1])?, [9, 7]);
        assert_eq!(x.sum_all(), 16);
    }
    Ok(())
}

#[test]
fn sum_4() -> Result<(), ZyxError> {
    {
        let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
        let x0 = x.relu().sum([-1])?;
        assert_eq!(x0, [8, 10, 18]);
    }
    Ok(())
}

#[test]
fn sum_5() -> Result<(), ZyxError> {
    {
        let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
        x = x.sum_all();
        debug_assert_eq!(x, [13i32]);
    }
    Ok(())
}

#[test]
fn max_1() -> Result<(), ZyxError> {
    {
        let x = Tensor::from([[4, 1, 3], [5, 2, 3], [6, 5, 7]]);
        let x0 = x.max([-1])?;
        let x1 = x.max([-2])?;
        let x2 = x.max_all();
        assert_eq!(x0, [4, 5, 7]);
        assert_eq!(x1, [6, 5, 7]);
        assert_eq!(x2, [7]);
    }
    Ok(())
}

#[test]
fn sum_large_2d() -> Result<(), ZyxError> {
    {
        // Large tensor that would benefit from loop splitting optimization
        let x = Tensor::from([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
            [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        ]);
        let x0 = x.sum([-1])?;
        assert_eq!(x0, [55, 155, 255, 355, 455, 555, 655, 755]);
    }
    Ok(())
}

#[test]
fn max_large_3d() -> Result<(), ZyxError> {
    {
        // Large 3D tensor that would benefit from loop splitting optimization
        let x = Tensor::from([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        ]);
        let x0 = x.max([-1])?;
        assert_eq!(x0, [[3, 6, 9], [12, 15, 18], [21, 24, 27]]);
    }
    Ok(())
}
