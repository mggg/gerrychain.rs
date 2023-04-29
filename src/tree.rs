//! Compiled spanning tree extensions for GerryChain.
use hashbrown::HashSet;
use std::mem;
use std::collections::VecDeque;

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::prelude::*;
use petgraph::visit::NodeIndexable;

use rustworkx::graph;
use rustworkx::tree::minimum_spanning_edges;

/// Helper function to allow reuse of spanning_tree object to reduce memory allocs
fn _minimum_spanning_tree(
    py: Python,
    graph: &graph::PyGraph,
    spanning_tree: &mut graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<()> {
    for edge in minimum_spanning_edges(py, graph, weight_fn, default_weight)?
        .edges
        .iter()
    {
        spanning_tree.add_edge(edge.0, edge.1, edge.2.clone_ref(py));
    }

    Ok(())
}

/// Bipartition tree by finding balanced cut edges of a spanning tree using
/// node contraction. Assumes that the tree is connected and is a spanning tree.
/// A balanced edge is defined as an edge that, when cut, will split the
/// population of the tree into two connected subtrees that have population near
/// the population target within some epsilon. The function returns a list of
/// all such possible cuts, represented as the set of nodes in one
/// partition/subtree. Wraps around ``_bipartition_tree``.
///
/// :param PyGraph graph: Spanning tree. Must be fully connected
/// :param pops: The populations assigned to each node in the graph.
/// :param float pop_target: The population target to reach when partitioning the
///     graph.
/// :param float epsilon: The maximum percent deviation from the pop_target
///     allowed while still being a valid balanced cut edge.
///
/// :returns: A list of tuples, with each tuple representing a distinct
///     balanced edge that can be cut. The tuple contains the root of one of the
///     two partitioned subtrees and the set of nodes making up that subtree.
#[pyfunction]
#[pyo3(text_signature = "(spanning_tree, pops, target_pop, epsilon)")]
pub fn bipartition_tree(
    spanning_tree: &graph::PyGraph,
    pops: Vec<f64>,
    pop_target: f64,
    epsilon: f64,
) -> Vec<(usize, Vec<usize>)> {
    _bipartition_tree(spanning_tree, pops, pop_target, epsilon)
}

/// Internal _bipartition_tree implementation.
fn _bipartition_tree(
    spanning_tree: &graph::PyGraph,
    pops: Vec<f64>,
    pop_target: f64,
    epsilon: f64,
) -> Vec<(usize, Vec<usize>)> {
    let mut pops = pops;
    let spanning_tree_graph = &spanning_tree.graph;
    let mut same_partition_tracker: Vec<Vec<usize>> =
        vec![Vec::new(); spanning_tree_graph.node_bound()]; // Keeps track of all all the nodes on the same side of the partition

    let mut node_queue: VecDeque<NodeIndex> = VecDeque::<NodeIndex>::new();
    for leaf_node in spanning_tree_graph.node_indices() {
        if spanning_tree_graph.neighbors(leaf_node).count() == 1 {
            node_queue.push_back(leaf_node);
        }
        same_partition_tracker[leaf_node.index()].push(leaf_node.index());
    }

    // BFS search for balanced nodes
    let mut balanced_nodes: Vec<(usize, Vec<usize>)> = vec![];
    let mut seen_nodes = HashSet::with_capacity(spanning_tree_graph.node_count());
    while !node_queue.is_empty() {
        let node = node_queue.pop_front().unwrap();
        if seen_nodes.contains(&node.index()) {
            continue;
        }

        // Mark as seen; push to queue if only one unseen neighbor
        let unseen_neighbors: Vec<NodeIndex> = spanning_tree
            .graph
            .neighbors(node)
            .filter(|node| !seen_nodes.contains(&node.index()))
            .collect();

        if unseen_neighbors.len() == 1 {
            // At leaf, will be false at root
            let pop = pops[node.index()];

            // Update neighbor pop
            let neighbor = unseen_neighbors[0];
            pops[neighbor.index()] += pop;

            // Check if balanced; mark as seen
            if pop >= pop_target * (1.0 - epsilon) && pop <= pop_target * (1.0 + epsilon) {
                balanced_nodes.push((node.index(), same_partition_tracker[node.index()].clone()));
            }
            seen_nodes.insert(node.index());

            // Update neighbor partition tracker
            let mut current_partition_tracker =
                mem::take(&mut same_partition_tracker[node.index()]);
            same_partition_tracker[neighbor.index()].append(&mut current_partition_tracker);

            // Queue neighbor
            node_queue.push_back(neighbor);
        } else if unseen_neighbors.is_empty() {
            break;  // Is root
        } 
    }

    balanced_nodes
}

/// Bipartition graph into two contiguous, population-balanced components using
/// mst. Assumes that the graph is contiguous. See :func:`~bipartition_tree` for
/// details on how balance is defined.
///
/// :param PyGraph graph: Undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. See
///     :func:`~minimum_spanning_tree` for details.
/// :param pops: The populations assigned to each node in the graph.
/// :param float pop_target: The population target to reach when partitioning
///     the graph.
/// :param float epsilon: The maximum percent deviation from the pop_target
///     allowed while still being a valid balanced cut edge.
///
/// :returns: A list of tuples, with each tuple representing a distinct
///     balanced edge that can be cut. The tuple contains the root of one of the
///     two partitioned subtrees and the set of nodes making up that subtree.
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, pops, target_pop, epsilon)")]
pub fn bipartition_graph_mst(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
    pops: Vec<f64>,
    pop_target: f64,
    epsilon: f64,
) -> PyResult<Vec<(usize, Vec<usize>)>> {
    let mut balanced_nodes: Vec<(usize, Vec<usize>)> = vec![];
    let mut mst = graph.clone();

    while balanced_nodes.is_empty() {
        mst.graph.clear_edges();
        _minimum_spanning_tree(py, graph, &mut mst, Some(weight_fn.clone()), 1.0)?;
        balanced_nodes = _bipartition_tree(&mst, pops.clone(), pop_target, epsilon);
    }

    Ok(balanced_nodes)
}