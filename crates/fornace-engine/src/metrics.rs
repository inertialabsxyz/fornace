//! Measurements: throughput, latency, and memory.

/// Returns the peak resident set size (RSS) of this process, in bytes.
///
/// The value is the largest RSS since the process started. It is not reset
/// between calls to `Engine::generate`.
pub(crate) fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage only writes into the struct we pass. A zeroed
    // rusage is a valid value for it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0;
    }
    let max_rss = usage.ru_maxrss as u64;
    // macOS reports ru_maxrss in bytes. Linux reports it in kilobytes.
    if cfg!(target_os = "macos") {
        max_rss
    } else {
        max_rss * 1024
    }
}
