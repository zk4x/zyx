// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The global runtime lock.

// Spinlock is better for debugging, but std::mutex::Mutex is better for release:
// the runtime lock is global, so a spinning waiter burns a core the holder needs.
#[derive(Debug)]
pub(crate) struct Mutex<T>(std::sync::Mutex<T>);

pub(crate) type MutexGuard<'a, T> = std::sync::MutexGuard<'a, T>;

impl<T> Mutex<T> {
    pub(super) const fn new(data: T) -> Self {
        Self(std::sync::Mutex::new(data))
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }

    /// Blocks, like `lock`: `Tensor::drop` reaches the lock through this name,
    /// and a non-blocking failure there has no recovery.
    pub(super) fn try_lock(&self) -> Result<MutexGuard<'_, T>, ()> {
        Ok(self.lock())
    }
}

// Standard spinlock, kept for debugging (swap in place of the block above).
/*use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
pub(crate) struct Mutex<T> {
    data: UnsafeCell<T>,
    lock: AtomicBool,
}

#[derive(Debug)]
pub(crate) struct MutexGuard<'a, T: 'a> {
    mutex: &'a Mutex<T>,
}

unsafe impl<T: Send> Sync for Mutex<T> {}
unsafe impl<T: Send> Send for Mutex<T> {}

unsafe impl<T: Sync> Sync for MutexGuard<'_, T> {}
unsafe impl<T: Send> Send for MutexGuard<'_, T> {}

impl<T> Mutex<T> {
    pub(super) const fn new(data: T) -> Self {
        Self { data: UnsafeCell::new(data), lock: AtomicBool::new(false) }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T> {
        loop {
            if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return MutexGuard { mutex: self };
            }

            while self.lock.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }

    pub(super) fn try_lock(&self) -> Result<MutexGuard<'_, T>, ()> {
        loop {
            if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return Ok(MutexGuard { mutex: self });
            }

            while self.lock.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }
}

impl<T> core::ops::Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<T> core::ops::DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.lock.store(false, Ordering::Release);
    }
}*/

// Good for debugging too
/*pub struct Mutex<T>(parking_lot::Mutex<T>);

impl<T> Mutex<T> {
    pub const fn new(data: T) -> Self {
        Self(parking_lot::Mutex::new(data))
    }

    pub fn lock(&self) -> parking_lot::MutexGuard<'_, T> {
        self.0.lock()
    }

    pub fn try_lock(&self) -> Result<parking_lot::MutexGuard<'_, T>, ()> {
        Ok(self.0.lock())
    }
}*/
