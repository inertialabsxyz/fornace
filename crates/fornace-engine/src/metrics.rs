//! Measurements: throughput, latency, and memory.
//!
//! "Memory" is the memory that the operating system charges to the process:
//!
//! - On macOS, the physical footprint (`phys_footprint`). Activity Monitor
//!   shows this value as "Memory". It includes Metal buffers. RSS does not,
//!   so on Apple silicon RSS can miss most of the memory that a model uses.
//! - On other systems, the resident set size (RSS).

/// Returns the memory that this process uses now, in bytes.
pub fn memory_bytes() -> u64 {
    imp::memory_bytes()
}

/// Returns the largest value of [`memory_bytes`] since the process started.
/// It is not reset between calls to `Engine::generate`.
pub fn peak_memory_bytes() -> u64 {
    imp::peak_memory_bytes()
}

#[cfg(target_os = "macos")]
mod imp {
    fn rusage() -> libc::rusage_info_v4 {
        // SAFETY: proc_pid_rusage writes one rusage_info_v4 into the buffer
        // for the flavor RUSAGE_INFO_V4. A zeroed struct is a valid value.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V4,
                (&raw mut info).cast(),
            )
        };
        // The call fails only for a bad pid or flavor, and both are fixed here.
        debug_assert_eq!(rc, 0);
        info
    }

    pub(super) fn memory_bytes() -> u64 {
        rusage().ri_phys_footprint
    }

    pub(super) fn peak_memory_bytes() -> u64 {
        rusage().ri_lifetime_max_phys_footprint
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn memory_bytes() -> u64 {
        memory_stats::memory_stats().map_or(0, |m| m.physical_mem as u64)
    }

    pub(super) fn peak_memory_bytes() -> u64 {
        // SAFETY: getrusage only writes into the struct we pass. A zeroed
        // rusage is a valid value for it.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return 0;
        }
        // Linux reports ru_maxrss in kilobytes.
        usage.ru_maxrss as u64 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touched_memory_shows_in_memory_and_peak() {
        const SIZE: usize = 256 << 20;
        let before = memory_bytes();
        let block = vec![1u8; SIZE];
        let during = memory_bytes();
        std::hint::black_box(&block);
        assert!(
            during >= before + SIZE as u64 * 9 / 10,
            "before {before}, during {during}"
        );
        assert!(peak_memory_bytes() >= during);
    }
}
