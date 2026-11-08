use crate::{l1_read32, l1_write32, publish};
use tt_isa::dataflow::{self, Action, Channel, Endpoint, Release, Stream};
use tt_isa::{dm, mailbox};

fn read(address: u64) -> u32 {
    unsafe { l1_read32(address) }
}
fn write(address: u64, value: u32) {
    unsafe { l1_write32(address, value) }
}

#[inline(never)]
pub fn check() -> Result<(), u32> {
    publish();
    let abort = read(dataflow::ABORT);
    if abort != 0 {
        return Err(abort);
    }
    for role in 0..3 {
        if read(mailbox::role::Mailbox::of(role).status()) == mailbox::status::PANICKED {
            abort_with(dm::error::ROLE);
            return Err(dm::error::ROLE);
        }
    }
    for base in [dm::MAILBOX_BASE, dm::nc::MAILBOX_BASE] {
        if read(base + dm::QUEUE_ERROR - dm::MAILBOX_BASE) != 0
            || read(base + mailbox::offset::STATUS) == mailbox::status::PANICKED
        {
            abort_with(dm::error::PEER);
            return Err(dm::error::PEER);
        }
    }
    Ok(())
}

/// Polls between full [`check`]s. Every poll still fences and reads the abort
/// word, so a cancellation is seen at once; a peer that panicked without
/// aborting is seen within this many polls.
const CHECK_EVERY: u32 = 64;

/// A credit wait's failure detection: the abort word on every poll, the full
/// [`check`] on the first and every [`CHECK_EVERY`]th. A countdown, not a
/// remainder: T2 has no remainder instruction.
#[derive(Default)]
pub struct Poll {
    until_check: u32,
}

impl Poll {
    pub const fn new() -> Self {
        Self { until_check: 0 }
    }

    #[inline(always)]
    pub fn tick(&mut self) -> Result<(), u32> {
        if self.until_check == 0 {
            self.until_check = CHECK_EVERY;
            check()?;
        } else {
            publish();
            let abort = read(dataflow::ABORT);
            if abort != 0 {
                return Err(abort);
            }
        }
        self.until_check -= 1;
        Ok(())
    }
}

pub fn abort_with(code: u32) {
    write(dataflow::ABORT, code);
    publish();
}

pub fn initialize() {
    let mut address = dataflow::INPUT;
    while address < dataflow::END {
        write(address, 0);
        address += 4;
    }
    publish();
}

#[inline(never)]
pub fn buffer(
    stream: Stream,
    action: Action,
    capacity: u16,
    endpoint: Endpoint,
    actor: u32,
) -> Result<(), u32> {
    let channel = Channel::of(stream, capacity).ok_or(dm::error::OP)?;
    if !channel.permits(endpoint, action) {
        return Err(dm::error::OP);
    }
    let reason = dataflow::WAIT_REASON + actor as u64 * 4;
    // Input 1..4, output 9..12, transfer 17..20 (`released`'s are 24..).
    let band = match stream {
        Stream::Input => 0,
        Stream::Output => 8,
        Stream::Transfer => 16,
    };
    write(reason, 1 + action.word() + band);
    let mut poll = Poll::new();
    loop {
        poll.tick()?;
        let received = read(channel.address) as u16;
        let consumed = read(channel.address + 4) as u16;
        let finished = match action {
            Action::Reserve => dataflow::room(received, consumed, capacity, 1),
            Action::Wait => dataflow::ready(received, consumed, 1),
            Action::Push => {
                if !dataflow::room(received, consumed, capacity, 1) {
                    return Err(dm::error::LENGTH);
                }
                write(channel.address, received.wrapping_add(1) as u32);
                true
            }
            Action::Pop => {
                if !dataflow::ready(received, consumed, 1) {
                    return Err(dm::error::LENGTH);
                }
                write(channel.address + 4, consumed.wrapping_add(1) as u32);
                true
            }
        };
        if finished {
            write(reason, 0);
            publish();
            return Ok(());
        }
    }
}

/// Wait until `target` output batches are packed by T2, or written out and
/// released by NC (the output counters' received or consumed count), or
/// `target` transfer batches are (the transfer counters' consumed count).
pub fn released(target: u16, which: Release) -> Result<(), u32> {
    let counter = match which {
        Release::Packed => dataflow::OUTPUT,
        Release::Written => dataflow::OUTPUT + 4,
        Release::Transferred => dataflow::TRANSFER + 4,
    };
    write(dataflow::WAIT_REASON, 24 + which.word());
    let mut poll = Poll::new();
    loop {
        poll.tick()?;
        if dataflow::reached(read(counter) as u16, target) {
            write(dataflow::WAIT_REASON, 0);
            return Ok(());
        }
    }
}

/// Record that `role` has retired `batch`, without waiting for its peers: the
/// region's last batch, which the mover's `KERNEL_WAIT` joins instead.
pub fn record_batch(role: u32, batch: u32) {
    write(dataflow::ROLE_PROGRESS + role as u64 * 4, batch);
    publish();
}

pub fn retire_batch(role: u32, batch: u32) -> Result<(), u32> {
    write(dataflow::WAIT_REASON + (role as u64 + 1) * 4, 32);
    record_batch(role, batch);
    let mut poll = Poll::new();
    loop {
        poll.tick()?;
        if (0..3).all(|peer| read(dataflow::ROLE_PROGRESS + peer * 4) >= batch) {
            write(dataflow::WAIT_REASON + (role as u64 + 1) * 4, 0);
            return Ok(());
        }
    }
}
