use std::iter::repeat;

pub(super) struct ProgressBar {
    count: u64,
    idx: u64,
}

impl ProgressBar {
    pub fn new(count: u64) -> Self {
        let n = 100;
        print!("[{}]", repeat('-').take(n).collect::<String>());
        Self { count, idx: 0 }
    }

    pub fn inc(&mut self, by: u64, message: &str) {
        self.idx = self.idx.saturating_add(by);
        let n = 100;
        let k = (self.idx * n) / self.count;
        print!(
            "\r                                                                                                    \r[{}{}] {message}",
            repeat('#').take(k as usize).collect::<String>(),
            repeat('-').take((n - k) as usize).collect::<String>(),
        );
        if self.idx == self.count {
            println!();
        }
    }
}
