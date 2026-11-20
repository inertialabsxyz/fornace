//! State-aware constrained sampling: a mask that permits only the commands
//! that the live state of an external system permits.
//!
//! The test case is a simulated payments ledger ([`ledger`]).

pub mod bench;
pub mod chat;
pub mod ledger;
pub mod range;
pub mod rng;
pub mod runs;
pub mod tasks;
pub mod trie;
