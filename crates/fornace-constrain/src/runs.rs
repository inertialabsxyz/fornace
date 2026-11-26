//! The run files of the bench binaries: the names, the task IDs, and a
//! writer that does not write over an earlier run.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::process::Command as Process;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};

/// A new file `dir/name`. An error if the file exists.
pub fn writer(dir: &Path, name: &str) -> Result<BufWriter<File>> {
    let path = dir.join(name);
    if path.exists() {
        bail!("{path:?} exists. Give another --tag");
    }
    eprintln!("writing {path:?}");
    Ok(BufWriter::new(File::create(&path)?))
}

/// `0-29,180,185-189` as a list of IDs.
pub fn parse_ids(text: &str) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((a, b)) => ids.extend(a.trim().parse::<usize>()?..=b.trim().parse::<usize>()?),
            None => ids.push(part.trim().parse()?),
        }
    }
    Ok(ids)
}

/// The short commit hash, with `-dirty` if the tree has changes.
pub fn commit() -> String {
    let run = |args: &[&str]| {
        Process::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let hash = run(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "nogit".to_string());
    let dirty =
        run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    if dirty { format!("{hash}-dirty") } else { hash }
}

/// The UTC date as `YYYY-MM-DD`.
pub fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0) as i64;
    civil_date(days)
}

/// The civil date of a day count from 1970-01-01 (H. Hinnant,
/// "chrono-compatible low-level date algorithms").
fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_dates() {
        assert_eq!(parse_ids("0-2,7").unwrap(), vec![0, 1, 2, 7]);
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(20_412), "2025-11-20");
        assert_eq!(civil_date(11_016), "2000-02-29");
    }
}
