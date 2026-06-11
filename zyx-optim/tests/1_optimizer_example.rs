use zyx_optim::Adam;

#[test]
fn optimizer_example() -> Result<(), zyx::ZyxError> {
    // Create optimizer with default parameters
    let mut optim = Adam {
        learning_rate: 0.001,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
        m: Vec::new(),
        v: Vec::new(),
        vm: Vec::new(),
        t: 0,
    };
    
    // Configure hyperparameters
    optim.learning_rate = 0.001;
    optim.betas = (0.9, 0.999);
    optim.eps = 1e-8;
    optim.weight_decay = 0.0;
    optim.amsgrad = false;
    
    println!("Optimizer initialized");
    println!("Learning rate: {}", optim.learning_rate);
    println!("Betas: {:?}", optim.betas);
    
    Ok(())
}
