//! GerryChain compiled extensions.
use pyo3::prelude::*;
use pyo3::Python;
use pyo3::wrap_pymodule;
use rustworkx::rustworkx as rustworkx_mod;

mod tree;
use tree::{bipartition_graph_mst, bipartition_tree};


/// GerryChain compiled extensions.
#[pymodule]
fn gerrychain_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bipartition_graph_mst, m)?)?;
    m.add_function(wrap_pyfunction!(bipartition_tree, m)?)?;
    m.add_wrapped(wrap_pymodule!(rustworkx_mod))?;
    Ok(())
}
