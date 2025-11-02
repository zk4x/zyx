//! Simple implementation of mutex based on spinlock.

use std::sync::MutexGuard;
pub struct Mutex<T>(std::sync::Mutex<T>);

impl<T> Mutex<T> {
    pub const fn new(data: T) -> Self {
        Self(std::sync::Mutex::new(data))
    }

    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().ok().unwrap()
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.0.try_lock().ok()
    }
}

// Spinlock is better for debugging, but std::mutex::Mutex is better for release
// Spinlock is also much faster
/*use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

// Standard spinlock, but will panic if it fails to lock after more than N tries
#[derive(Debug)]
pub struct Mutex<T> {
    data: UnsafeCell<T>,
    lock: AtomicBool,
}

#[derive(Debug)]
pub struct MutexGuard<'a, T: 'a> {
    lock: &'a AtomicBool,
    data: &'a UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for Mutex<T> {}
unsafe impl<T: Send> Send for Mutex<T> {}

unsafe impl<T: Sync> Sync for MutexGuard<'_, T> {}
unsafe impl<T: Send> Send for MutexGuard<'_, T> {}

impl<T> Mutex<T> {
    pub(super) const fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            lock: AtomicBool::new(false),
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T> {
        const N: usize = 1_000_000_000;

        let mut i = 0;
        loop {
            if self
                .lock
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return MutexGuard {
                    lock: &self.lock,
                    data: &self.data,
                };
            }

            while self.lock.load(Ordering::Relaxed) {
                //core::sync::atomic::spin_loop_hint();
                //std::thread::sleep(std::time::Duration::from_secs(1));
                core::hint::spin_loop();
                i += 1;
                debug_assert!(i > N, "Failed to unlock mutex after {N} tries. Panicking in order to avoid deadlock.");
            }
            /*debug_assert!(
                i > N,
                "Failed to unlock mutex after million tries. Panicking in order to avoid deadlock."
            );*/
        }
    }

    pub(super) fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        const N: usize = 1_000;

        let mut i = 0;
        loop {
            if self
                .lock
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return Some(MutexGuard {
                    lock: &self.lock,
                    data: &self.data,
                });
            }

            while self.lock.load(Ordering::Relaxed) {
                //core::sync::atomic::spin_loop_hint();
                //std::thread::sleep(std::time::Duration::from_secs(1));
                core::hint::spin_loop();
                i += 1;
                if i > N {
                    return None;
                }
            }
        }
    }
}

impl<T> core::ops::Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.data.get() }
    }
}

impl<T> core::ops::DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.store(false, Ordering::Release);
    }
}*/
