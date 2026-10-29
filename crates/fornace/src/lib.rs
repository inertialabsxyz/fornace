//! Fornace: small models that run where your data lives.
//!
//! Fornace is an ongoing project on local AI inference in Rust. It builds
//! inference engines down to the kernels and measures each change. The
//! results go in `results/`.
//!
//! This crate is the root of the workspace. It exports the engine as
//! [`engine`].

pub use fornace_engine as engine;
