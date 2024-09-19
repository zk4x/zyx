//! Simple implementation of mutex based on spinlock.

/*use std::sync::MutexGuard;

pub(super) struct Mutex<T>(std::sync::Mutex<T>);

impl<T> Mutex<T> {
    pub(super) const fn new(data: T) -> Self {
        Self(std::sync::Mutex::new(data))
    }

    pub(crate) fn lock(&self) -> MutexGuard<T> {
        self.0.lock().ok().unwrap()
    }
}*/

use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

// Standard spinlock, but will panic if it fails to lock after more than N tries
pub(super) struct Mutex<T, const N: usize> {
    data: UnsafeCell<T>,
    lock: AtomicBool,
}

pub(super) struct MutexGuard<'a, T: 'a> {
    lock: &'a AtomicBool,
    data: &'a UnsafeCell<T>,
}

unsafe impl<T: Send, const N: usize> Sync for Mutex<T, N> {}
unsafe impl<T: Send, const N: usize> Send for Mutex<T, N> {}

unsafe impl<T: Sync> Sync for MutexGuard<'_, T> {}
unsafe impl<T: Send> Send for MutexGuard<'_, T> {}

impl<T, const N: usize> Mutex<T, N> {
    pub(super) const fn new(data: T) -> Self {
        return Self {
            data: UnsafeCell::new(data),
            lock: AtomicBool::new(false),
        };
    }

    #[inline(always)]
    pub(super) fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
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
                core::hint::spin_loop();
                if i > N {
                    panic!("Failed to unlock mutex after million tries. Panicking in order to avoid deadlock.");
                }
                i += 1;
            }
            if i > N {
                panic!("Failed to unlock mutex after million tries. Panicking in order to avoid deadlock.");
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
        self.lock.store(false, Ordering::Release)
    }
}
