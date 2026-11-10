//! A role program with its block repeats: the instructions as the program
//! slot holds them, and `(start, len, count)` loops the role runner expands
//! as it pushes (`tt_isa::mailbox::loops`: a header stored with the program).
//!
//! How a body too long for the 32-entry replay buffer (`frontend::REPLAY`)
//! runs without being unrolled into the 8192-word program slot: an SFPU row
//! loop of 300 instructions is stored once and pushed 32 times. Nothing here
//! is an instruction of its own -- the loops are a header stored with the
//! program, flagged in its length -- and what reaches the coprocessor is exactly
//! [`Code::expand`]'s stream, which is what every model and gate runs.

use tt_isa::isa::Instruction;
use tt_isa::mailbox::loops;

/// `ins[start..start + len]` pushed `count` times where it is stored once.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Loop {
    pub start: u32,
    pub len: u32,
    pub count: u32,
}

impl Loop {
    fn end(self) -> u32 {
        self.start + self.len
    }
}

/// Why a program's loops cannot be handed to the runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopError {
    /// More than `mailbox::loops::MAX`.
    TooMany(usize),
    /// A field does not fit `mailbox::loops::entry`, or a loop runs past the
    /// program.
    Field(Loop),
    /// Two loops overlap without one holding the other, or one is three deep.
    Nesting(Loop, Loop),
}

impl std::fmt::Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopError::TooMany(n) => {
                write!(f, "{n} block repeats, past the runner's {}", loops::MAX)
            }
            LoopError::Field(l) => write!(f, "block repeat {l:?} does not fit the runner's table"),
            LoopError::Nesting(a, b) => write!(f, "block repeats {a:?} and {b:?} neither disjoint nor nested once"),
        }
    }
}

impl std::error::Error for LoopError {}

/// Instructions and the loops over them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Code {
    pub ins: Vec<Instruction>,
    pub loops: Vec<Loop>,
}

impl Code {
    /// The plain program: no loops.
    pub fn plain(ins: Vec<Instruction>) -> Self {
        Code { ins, loops: Vec::new() }
    }

    /// The stream the runner pushes: each loop's range `count` times, as
    /// `tt_firmware::corpus::Pusher::span` walks it.
    pub fn expand(&self) -> Vec<Instruction> {
        let mut out = Vec::new();
        span(&self.ins, &self.loops, 0, self.ins.len() as u32, &mut out);
        out
    }

    /// The program as its slot stores it and the length word that names it
    /// (`mailbox::loops`): with loops, a header -- their count, their entries
    /// -- before the code, and `LOOPED` set in the length; without, the code
    /// alone. Checked as the runner checks it.
    pub fn stored(&self) -> Result<(Vec<u32>, u32), LoopError> {
        let words: Vec<u32> = self.ins.iter().map(|i| i.word()).collect();
        stored(&words, &self.loops)
    }
}

/// [`Code::stored`] for a program that is already its instruction words, which
/// is what the native backend hands over: codegen encodes, nothing decodes.
pub fn stored(words: &[u32], loops: &[Loop]) -> Result<(Vec<u32>, u32), LoopError> {
    if loops.is_empty() {
        let n = words.len() as u32;
        return Ok((words.to_vec(), n));
    }
    check(loops, words.len() as u32)?;
    let mut out = vec![loops.len() as u32];
    for l in loops {
        out.push(loops::entry(l.start, l.len, l.count).ok_or(LoopError::Field(*l))?);
    }
    out.extend_from_slice(words);
    let n = out.len() as u32;
    Ok((out, n | loops::LOOPED))
}

/// The runner's rules: at most `MAX` loops, inside the program, any two
/// disjoint or one inside the other, at most two deep.
pub fn check(ls: &[Loop], len: u32) -> Result<(), LoopError> {
    if ls.len() > loops::MAX {
        return Err(LoopError::TooMany(ls.len()));
    }
    for &l in ls {
        if l.end() > len || loops::entry(l.start, l.len, l.count).is_none() {
            return Err(LoopError::Field(l));
        }
    }
    for &a in ls {
        let mut depth = 0;
        for &b in ls {
            if a == b {
                continue;
            }
            let disjoint = a.end() <= b.start || b.end() <= a.start;
            let inside = b.start <= a.start && a.end() <= b.end() && (a.start, a.len) != (b.start, b.len);
            let around = a.start <= b.start && b.end() <= a.end() && (a.start, a.len) != (b.start, b.len);
            if !(disjoint || inside || around) {
                return Err(LoopError::Nesting(a, b));
            }
            depth += usize::from(inside);
        }
        if depth > 1 {
            return Err(LoopError::Nesting(a, a));
        }
    }
    Ok(())
}

fn span(ins: &[Instruction], ls: &[Loop], lo: u32, hi: u32, out: &mut Vec<Instruction>) {
    let mut at = lo;
    loop {
        let next = ls
            .iter()
            .filter(|l| l.start >= at && l.end() <= hi && (l.start, l.end()) != (lo, hi))
            .min_by_key(|l| (l.start, std::cmp::Reverse(l.len)))
            .copied();
        let Some(l) = next else {
            out.extend_from_slice(&ins[at as usize..hi as usize]);
            return;
        };
        out.extend_from_slice(&ins[at as usize..l.start as usize]);
        for _ in 0..l.count {
            span(ins, ls, l.start, l.end(), out);
        }
        at = l.end();
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use tt_isa::isa::generated::encode;

    fn ins(n: u32) -> Vec<Instruction> {
        (0..n).map(|k| encode::sfploadi(0, 2, k).unwrap()).collect()
    }

    fn imm(v: &[Instruction]) -> Vec<u32> {
        v.iter().map(|i| i.operand("Imm16").unwrap()).collect()
    }

    #[test]
    fn nested_loops_expand_as_the_runner_pushes() {
        let c = Code { ins: ins(8), loops: vec![Loop { start: 1, len: 5, count: 2 }, Loop { start: 2, len: 2, count: 3 }] };
        assert_eq!(imm(&c.expand()), [0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 1, 2, 3, 2, 3, 2, 3, 4, 5, 6, 7]);
        let (words, len) = c.stored().unwrap();
        assert_eq!(len, 11 | loops::LOOPED, "two entries, their count, eight words");
        assert_eq!(words[0], 2);
    }

    #[test]
    fn what_the_runner_would_refuse_is_refused() {
        let over = Loop { start: 2, len: 4, count: 2 };
        let bad = [
            vec![Loop { start: 0, len: 3, count: 2 }, Loop { start: 2, len: 3, count: 2 }],
            vec![Loop { start: 0, len: 9, count: 2 }],
            vec![Loop { start: 0, len: 6, count: 2 }, over, Loop { start: 3, len: 1, count: 2 }],
            vec![Loop { start: 0, len: 1, count: 129 }],
        ];
        for ls in bad {
            assert!(check(&ls, 8).is_err(), "{ls:?}");
        }
    }
}
