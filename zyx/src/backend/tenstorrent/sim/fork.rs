//! Running simulator work in a forked child, so that a fatal error is reportable.
//!
//! `libttsim` terminates the process on any contract violation. Without isolation
//! that means the test runner disappears mid-run, printing libttsim's message and
//! nothing else — no test name, no backtrace, no indication of which of forty tests
//! was responsible.
//!
//! [`fork_scope`] runs the risky part in a child process. A fatal error kills the
//! child; the parent observes the exit status and turns it into an ordinary
//! assertion failure that names the test.
//!
//! This is the isolation mechanism ttsim's own documentation recommends, and it is
//! also how it suggests reusing expensive setup: initialize once, fork per case.

use std::io;

/// Why a [`fork_scope`] child did not complete normally.
#[derive(Debug)]
pub enum ForkError {
    /// `fork` itself failed.
    Fork(io::Error),
    /// `waitpid` failed.
    Wait(io::Error),
    /// The child ran to completion but exited non-zero — normally because the
    /// closure panicked, or because libttsim called `_Exit`.
    Exited(i32),
    /// The child was killed by a signal.
    Signalled(i32),
}

impl std::fmt::Display for ForkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForkError::Fork(e) => write!(f, "fork failed: {e}"),
            ForkError::Wait(e) => write!(f, "waitpid failed: {e}"),
            ForkError::Exited(PANIC_EXIT_CODE) => {
                write!(f, "the forked child panicked; its message is above")
            }
            ForkError::Exited(code) => write!(
                f,
                "the forked child exited with status {code}. libttsim terminates the \
                 process via _Exit on any contract violation, so its diagnostic is \
                 above this line."
            ),
            ForkError::Signalled(sig) => {
                write!(f, "the forked child was killed by signal {sig}")
            }
        }
    }
}

impl std::error::Error for ForkError {}

/// Exit code used when the closure panics, to distinguish that from a libttsim
/// `_Exit` (which uses 1).
const PANIC_EXIT_CODE: i32 = 101;

/// Held exclusively for the instant of every `fork`, and shared by
/// [`outside_fork`] while a parent thread does library work.
///
/// A forked child inherits only the thread that forked, but every lock in the
/// address space as it stood: a lock another thread held at that instant is
/// held forever in the child, and the first thing in the child to take it
/// waits forever. `cargo test` runs tests on several threads at once, so a
/// test computing its `burn-flex` reference in the parent while another test
/// forks is exactly that. Seen once in the end-to-end tier: a training gate's
/// child waiting on a futex, its server thread idle, for ten minutes; every
/// test passed alone.
static GATE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Run `f` in the parent -- library work a test does outside any
/// [`fork_scope`], such as computing a host reference -- so that no other
/// thread forks while it runs. Any number may run at once; a fork waits for all
/// of them.
pub fn outside_fork<R>(f: impl FnOnce() -> R) -> R {
    let _shared = GATE.read().unwrap_or_else(|p| p.into_inner());
    f()
}

/// Run `f` in a forked child process and wait for it.
///
/// Returns `Ok(())` if the child completed without panicking, and a [`ForkError`]
/// otherwise. The closure's return value is deliberately not propagated: it lives
/// in another address space. Communicate results by asserting inside the closure,
/// or by writing to a file or pipe the parent can read.
///
/// # Caveats
///
/// Fork at a quiescent point — with no `libttsim` call in progress. Since the
/// simulator only advances inside `clock`, any point in ordinary sequential code
/// qualifies. Library work in the parent that may run while another test forks
/// goes through [`outside_fork`].
///
/// The child inherits the parent's buffered stdio, so both streams are flushed
/// before forking to avoid duplicated output.
pub fn fork_scope<F: FnOnce()>(f: F) -> Result<(), ForkError> {
    use std::io::Write;

    // Anything still buffered would otherwise be written twice: once by the child
    // and once by the parent.
    let _ = io::stdout().flush();
    let _ = io::stderr().flush();

    // No thread is inside `outside_fork` while this is held, so no library
    // lock those threads take is held across the fork.
    let exclusive = GATE.write().unwrap_or_else(|p| p.into_inner());
    // SAFETY: fork has no preconditions. The child may still meet a lock held
    // at fork time by a thread doing library work *outside* `outside_fork`
    // (see `GATE`), which is why parent-side work goes through it.
    let pid = unsafe { libc::fork() };
    // Both processes release their own copy: the child's is the forking
    // thread's, which is the child's only thread.
    drop(exclusive);

    match pid {
        -1 => Err(ForkError::Fork(io::Error::last_os_error())),
        0 => {
            // Child. Never return from here: unwinding past a fork point would run
            // the parent's test harness a second time in this address space.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let _ = io::stdout().flush();
            let _ = io::stderr().flush();
            let code = if result.is_ok() { 0 } else { PANIC_EXIT_CODE };
            // `_exit`, not `exit`: the child must not run atexit handlers or flush
            // the parent's inherited stdio state a second time.
            // SAFETY: _exit is always safe to call and does not return.
            unsafe { libc::_exit(code) }
        }
        pid => {
            let mut status: libc::c_int = 0;
            // SAFETY: `pid` is a live child of this process; `status` is a valid
            // pointer to an int.
            let rc = unsafe { libc::waitpid(pid, &mut status, 0) };
            if rc == -1 {
                return Err(ForkError::Wait(io::Error::last_os_error()));
            }
            if libc::WIFEXITED(status) {
                match libc::WEXITSTATUS(status) {
                    0 => Ok(()),
                    code => Err(ForkError::Exited(code)),
                }
            } else if libc::WIFSIGNALED(status) {
                Err(ForkError::Signalled(libc::WTERMSIG(status)))
            } else {
                Err(ForkError::Exited(-1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hazard `GATE` exists for: a lock another thread holds at the
    /// instant of `fork` is held forever in the child. With the work that
    /// holds it inside `outside_fork`, the fork waits for it, and the child
    /// takes the lock at once. (Without the gate the child would wait forever;
    /// the alarm turns that into a failure rather than a hung suite.)
    #[test]
    fn a_lock_held_by_parent_work_is_not_inherited_by_the_child() {
        use std::sync::{mpsc, Mutex};
        static LIBRARY: Mutex<()> = Mutex::new(());
        let (held, wait) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            outside_fork(|| {
                let _lock = LIBRARY.lock().unwrap();
                held.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(200));
            })
        });
        wait.recv().unwrap();
        let r = fork_scope(|| {
            // SAFETY: arms a signal for this child only; nothing else uses it.
            unsafe { libc::alarm(5) };
            drop(LIBRARY.lock().unwrap());
        });
        worker.join().unwrap();
        r.expect("the child must get the lock the parent's work released");
    }

    #[test]
    fn clean_child_succeeds() {
        fork_scope(|| {}).expect("an empty closure should succeed");
    }

    #[test]
    fn panicking_child_is_reported_not_propagated() {
        let err = fork_scope(|| panic!("deliberate")).expect_err("a panic should be reported");
        assert!(matches!(err, ForkError::Exited(PANIC_EXIT_CODE)), "{err:?}");
        assert!(err.to_string().contains("panicked"));
    }

    #[test]
    fn hard_exit_in_child_is_reported() {
        // Stands in for libttsim's `_Exit` on a contract violation: no unwinding,
        // no destructors. The parent must survive and report it.
        let err =
            fork_scope(|| unsafe { libc::_exit(1) }).expect_err("a hard exit should be reported");
        assert!(matches!(err, ForkError::Exited(1)), "{err:?}");
        assert!(err.to_string().contains("libttsim"));
    }

    #[test]
    fn fatal_signal_in_child_is_reported() {
        let err = fork_scope(|| unsafe {
            libc::raise(libc::SIGKILL);
        })
        .expect_err("a killed child should be reported");
        assert!(
            matches!(err, ForkError::Signalled(libc::SIGKILL)),
            "{err:?}"
        );
    }

    #[test]
    fn parent_state_is_untouched_by_the_child() {
        // The child gets a copy-on-write copy, so mutations must not be visible.
        let mut counter = 0u32;
        fork_scope(|| {
            // This mutation happens in the child's address space only.
            let mut local = counter;
            local += 100;
            std::hint::black_box(local);
        })
        .unwrap();
        counter += 1;
        assert_eq!(counter, 1);
    }
}
