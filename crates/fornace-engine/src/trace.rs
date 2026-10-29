//! Spans that measure GPU work correctly.
//!
//! Candle queues GPU work and returns before the work is done. A plain span
//! then measures only the time to queue the work. A `DeviceSpan` can wait for
//! the device when it starts and when it ends, so that the span contains the
//! GPU time of its own work and no other work.

use candle_core::{Device, Result};
use tracing::span::EnteredSpan;

/// Enters an INFO `DeviceSpan` on `$device`. The span has the given name and
/// fields, plus the field `gpu_delta_bytes`.
macro_rules! device_span {
    (@level $level:expr, $device:expr, $sync:expr, $name:literal $(, $($fields:tt)*)?) => {
        $crate::trace::DeviceSpan::enter(
            tracing::span!(
                $level,
                $name,
                gpu_delta_bytes = tracing::field::Empty
                $(, $($fields)*)?
            ),
            $device,
            $sync,
        )
    };
    ($device:expr, $sync:expr, $name:literal $(, $($fields:tt)*)?) => {
        $crate::trace::device_span!(@level tracing::Level::INFO, $device, $sync, $name $(, $($fields)*)?)
    };
}
pub(crate) use device_span;

/// Enters a DEBUG `DeviceSpan`. Use it for spans inside the model, which run
/// many times for each token. The default trace filter turns them off.
macro_rules! device_debug_span {
    ($($args:tt)*) => {
        $crate::trace::device_span!(@level tracing::Level::DEBUG, $($args)*)
    };
}
pub(crate) use device_debug_span;

/// An entered span that, when `sync` is true, waits for the device on entry
/// and on drop. On drop it records the change in GPU allocated bytes in the
/// field `gpu_delta_bytes`. A span that the filter turns off does nothing:
/// it does not wait and does not read the GPU memory.
pub(crate) struct DeviceSpan {
    span: EnteredSpan,
    device: Device,
    sync: bool,
    gpu_start: u64,
}

impl DeviceSpan {
    pub(crate) fn enter(span: tracing::Span, device: &Device, sync: bool) -> Result<DeviceSpan> {
        let enabled = !span.is_disabled();
        let sync = sync && enabled;
        // Wait before the span starts, so earlier GPU work is not counted here.
        if sync {
            device.synchronize()?;
        }
        Ok(DeviceSpan {
            gpu_start: if enabled {
                gpu_allocated_bytes(device)
            } else {
                0
            },
            span: span.entered(),
            device: device.clone(),
            sync,
        })
    }
}

impl Drop for DeviceSpan {
    fn drop(&mut self) {
        // Wait while the span is still entered, so its GPU work is counted
        // here. Drop cannot return an error, and a failed wait only makes
        // this one measurement wrong, so we ignore it.
        if self.sync {
            let _ = self.device.synchronize();
        }
        if self.span.is_disabled() {
            return;
        }
        let delta = gpu_allocated_bytes(&self.device) as i64 - self.gpu_start as i64;
        self.span.record("gpu_delta_bytes", delta);
    }
}

/// Returns the bytes that Metal has allocated on `device`, or 0 on the CPU.
pub(crate) fn gpu_allocated_bytes(device: &Device) -> u64 {
    match device {
        #[cfg(feature = "metal")]
        Device::Metal(m) => m.device().current_allocated_size() as u64,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_span_waits_without_error_and_has_no_gpu_bytes() {
        let device = Device::Cpu;
        let span = DeviceSpan::enter(tracing::info_span!("test"), &device, true).unwrap();
        assert_eq!(span.gpu_start, 0);
        assert_eq!(gpu_allocated_bytes(&device), 0);
    }
}
