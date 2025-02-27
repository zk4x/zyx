use std::{io::Write, iter::repeat};

pub(super) struct ProgressBar {
    count: u64,
    idx: u64,
}

impl ProgressBar {
    pub fn new(count: u64) -> Self {
        let n = 5;
        print!("[{}]", repeat('-').take(n).collect::<String>());
        Self { count, idx: 0 }
    }

    pub fn inc(&mut self, by: u64, message: &str) {
        self.idx = self.idx.saturating_add(by);
        //let n = 5;
        //let k = (self.idx * n) / self.count;
        print!(
            "\r[{0:1$}/{2}] {message}",
            self.idx,
            (self.count as f32).log10() as usize + 1,
            self.count,
            //repeat('=').take(k as usize).collect::<String>(),
            //repeat(' ').take((n - k) as usize).collect::<String>(),
            //repeat(' ').take((n) as usize).collect::<String>(),
        );

        _ = std::io::stdout().flush();

        if self.idx == self.count {
            println!();
        }
    }
}
