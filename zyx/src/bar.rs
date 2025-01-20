use std::iter::repeat;

pub(super) struct ProgressBar {
    count: u64,
    idx: u64,
}

impl ProgressBar {
    pub fn new(count: u64) -> Self {
        let n = 20;
        println!("[{}]", repeat('-').take(n).collect::<String>());
        Self { count, idx: 0 }
    }

    pub fn inc(&mut self, by: u64, message: &str) {
        self.idx = self.idx.saturating_add(by);
        let n = 20;
        let k = self.idx / self.count * n;
        println!(
            "[{}{}] {message}",
            repeat('-').take(k as usize).collect::<String>(),
            repeat('-').take((n - k) as usize).collect::<String>(),
        );
    }
}
