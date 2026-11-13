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

/// Returns the peak memory of another process, as [`peak_memory_bytes`]
/// gives it for this process. `None` if the process does not exist, or if
/// this process cannot read it.
pub fn peak_memory_bytes_of(pid: u32) -> Option<u64> {
    imp::peak_memory_bytes_of(pid)
}

#[cfg(target_os = "macos")]
mod imp {
    fn rusage(pid: u32) -> Option<libc::rusage_info_v4> {
        // SAFETY: proc_pid_rusage writes one rusage_info_v4 into the buffer
        // for the flavor RUSAGE_INFO_V4. A zeroed struct is a valid value.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                pid as libc::c_int,
                libc::RUSAGE_INFO_V4,
                (&raw mut info).cast(),
            )
        };
        (rc == 0).then_some(info)
    }

    fn own() -> libc::rusage_info_v4 {
        // The call fails only for a bad pid or flavor, and both are fixed here.
        rusage(std::process::id()).expect("proc_pid_rusage of this process")
    }

    pub(super) fn memory_bytes() -> u64 {
        own().ri_phys_footprint
    }

    pub(super) fn peak_memory_bytes() -> u64 {
        own().ri_lifetime_max_phys_footprint
    }

    pub(super) fn peak_memory_bytes_of(pid: u32) -> Option<u64> {
        rusage(pid).map(|info| info.ri_lifetime_max_phys_footprint)
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
            return memory_bytes();
        }
        // Linux reports ru_maxrss in kilobytes. The kernel reads it from an
        // approximate per-CPU counter, so it can lag behind the RSS that
        // /proc reports. The peak is never less than the current value.
        (usage.ru_maxrss as u64 * 1024).max(memory_bytes())
    }

    /// `VmHWM` of `/proc/<pid>/status`: the peak RSS.
    pub(super) fn peak_memory_bytes_of(pid: u32) -> Option<u64> {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
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

    #[test]
    fn the_peak_of_a_child_process_is_readable() {
        let mut child = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let peak = peak_memory_bytes_of(child.id());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(peak.is_some_and(|p| p > 0), "peak {peak:?}");
        assert_eq!(peak_memory_bytes_of(u32::MAX / 2), None);
    }
}
