use zyx::{Tensor, ZyxError};

#[test]
fn reshape_1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    x = x.reshape([8, 1])?;
    x = x.reshape([1, 2, 1, 4])?;
    x = x.reshape([4, 2])?;
    assert_eq!(x, [[4, 5], [2, 1], [3, 4], [1, 4]]);
    Ok(())
}

#[test]
fn reshape_permute_1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    x = x.reshape([8, 1])?;
    x = x.reshape([1, 2, 1, 4])?.permute([2, 3, 1, 0])?;
    x = x.reshape([4, 2])?.exp2().cast(zyx::DType::I32);
    assert_eq!(x, [[16, 8], [32, 16], [4, 2], [2, 16]]);
    Ok(())
}

#[test]
fn expand_1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]).reshape([1, 1, 1, 4])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 4, 1])?;
    let c = a + b;
    assert_eq!(c, [[[[6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12]]]]);
    Ok(())
}

#[test]
fn permute_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4, 5, 2, 1], [3, 4, 1, 4]]);
    let y = x.permute([1, 0])?;
    assert_eq!(y, [[4, 3], [5, 4], [2, 1], [1, 4]]);
    Ok(())
}

#[test]
fn pad_1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let c = a.pad_zeros([(0, 0), (0, 2)])?;
    assert_eq!(c, [[1, 2], [3, 4], [0, 0], [0, 0]]);
    Ok(())
}

#[test]
fn pad_2() -> Result<(), ZyxError> {
    let a = Tensor::from([[1i32, 2], [3, 4]]).reshape([1, 1, 2, 2])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 1, 4])?;
    let c = a.pad_zeros([(0, 2), (0, 2)])? + b;
    assert_eq!(c, [[[[6i32, 8, 7, 8], [8, 10, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]]]);
    Ok(())
}

#[test]
fn rope_1() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5, 6, 7, 8]).reshape([2, 4])?;
    let sin_freq = Tensor::from([[2, 3], [3, 1]]);
    let cos_freq = Tensor::from([[2, 3], [3, 1]]);

    let a = x.pad_zeros([(-2, 0)])?;
    let b = -x.pad_zeros([(0, -2)])?;
    let z = &a * &sin_freq - &b * &cos_freq;
    let z2 = a * sin_freq + b * cos_freq;
    let z = z.pad_zeros([(0, 2)])? + z2.pad_zeros([(2, 0)])?;
    assert_eq!(z, [[8, 18, 4, 6], [36, 14, 6, 2]]);
    Ok(())
}

#[test]
fn rope_2() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5, 6, 7, 8]).reshape([1, 2, 4])?;
    let base = 10000f32;

    let [batch_size, seq_len, embed_dim] = x.dims()?;

    assert_eq!(embed_dim % 2, 0, "Embedding dimension should be even for RoPE.");

    // Generate the position indices
    let position = Tensor::arange(0., seq_len as f32, 1.)?.unsqueeze(1)?; // Shape: (seq_len, 1)

    // Create a tensor of frequencies for each dimension
    let mut freqs = Tensor::arange(0., embed_dim as f32 / 2., 1.)?; // Shape: (embed_dim // 2)
    freqs = Tensor::from(base).pow(freqs * (2 / embed_dim) as f32)?; // Apply scaling for frequency

    // Create the positional encoding matrix (sinusoidal)
    let pos_enc = position * freqs; // Shape: (seq_len, embed_dim // 2)
    //println!("{pos_enc}");

    // Apply sin and cos to each dimension
    let sin_enc = pos_enc.sin(); // Shape: (seq_len, embed_dim // 2)
    let cos_enc = pos_enc.cos(); // Shape: (seq_len, embed_dim // 2)
    //Tensor::realize([&sin_enc, &cos_enc])?;

    // Now, interleave sin and cos values for the full embedding (pairing them)
    // sin_enc -> even dimensions, cos_enc -> odd dimensions
    let sin_enc = sin_enc.unsqueeze(0)?.expand_axis(0, batch_size)?; // Expand for batch size
    let cos_enc = cos_enc.unsqueeze(0)?.expand_axis(0, batch_size)?; // Expand for batch size
    //Tensor::realize([&sin_enc, &cos_enc])?;
    println!("{sin_enc}\n{cos_enc}");
    panic!();

    // Combine sin and cos to create the final embedding
    // The idea is to apply sin/cos to even and odd dimensions
    let x = x.rope(sin_enc, cos_enc)?;

    println!("{x}");

    Ok(())
}
