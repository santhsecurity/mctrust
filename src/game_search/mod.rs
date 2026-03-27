//! Environment-based MCTS — the classic game-tree search engine.
//!
//! Implements the four MCTS phases: Selection → Expansion → Simulation → Backpropagation.

use rand::SeedableRng;

use crate::config::SearchConfig;
use crate::environment::{Environment, GameState};
use crate::node::{Node, NodeStats};

/// Game-tree MCTS engine.
///
/// Given an [`Environment`], searches for the best action by running
/// repeated simulations through a growing tree of explored states.
///
/// # Type Parameters
///
/// - `E`: the environment type implementing [`Environment`].
///
/// # Examples
///
/// ```
/// use mctrust::{Environment, GameState, GameSearch, SearchConfig, Reward};
///
/// #[derive(Clone)]
/// struct Counter { value: i32, target: i32 }
///
/// #[derive(Clone, Debug, PartialEq)]
/// enum Step { Inc, Dec }
///
/// impl Environment for Counter {
///     type Action = Step;
///
///     fn legal_actions(&self) -> Vec<Step> {
///         vec![Step::Inc, Step::Dec]
///     }
///
///     fn apply(&mut self, action: &Step) {
///         match action {
///             Step::Inc => self.value += 1,
///             Step::Dec => self.value -= 1,
///         }
///     }
///
///     fn evaluate(&self) -> GameState {
///         if self.value == self.target {
///             GameState::Win(Reward::WIN)
///         } else if (self.value - self.target).abs() > 10 {
///             GameState::Loss
///         } else {
///             GameState::Ongoing
///         }
///     }
/// }
///
/// let game = Counter { value: 0, target: 3 };
/// let config = SearchConfig::builder().iterations(1_000).max_depth(15).build();
/// let mut search = GameSearch::new(game, config);
///
/// if let Some(best) = search.run() {
///     // best is the action with most visits from root
///     println!("Best action: {best:?}");
/// }
/// let _all_moves = vec![Step::Inc, Step::Dec];
/// ```
pub struct GameSearch<E: Environment> {
    /// The root environment state.
    pub(crate) root_env: E,

    /// Search hyperparameters.
    pub(crate) config: SearchConfig,

    /// Flat arena of tree nodes. Index 0 is always root.
    pub(crate) nodes: Vec<Node<E::Action>>,

    /// Deterministic RNG for simulation rollouts.
    pub(crate) rng: rand::rngs::StdRng,
}

/// Serialisable game-search checkpoint used for mid-search persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameSearchCheckpoint<E>
where
    E: Environment + Clone,
    E::Action: serde::Serialize + for<'action> serde::Deserialize<'action>,
{
    /// Root environment at checkpoint time.
    pub root_env: E,
    /// Search configuration at checkpoint time.
    pub config: SearchConfig,
    /// Full search tree snapshot.
    pub nodes: Vec<Node<E::Action>>,
}

impl<E: Environment> GameSearch<E> {
    /// Creates a new search engine from an environment and configuration.
    #[must_use]
    pub fn new(environment: E, config: SearchConfig) -> Self {
        let root_actions = environment.legal_actions();
        let root = Node::root(root_actions);

        Self {
            root_env: environment,
            config,
            nodes: vec![root],
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Creates a new search engine with a fixed seed for reproducible results.
    #[must_use]
    pub fn with_seed(environment: E, config: SearchConfig, seed: u64) -> Self {
        let root_actions = environment.legal_actions();
        let root = Node::root(root_actions);

        Self {
            root_env: environment,
            config,
            nodes: vec![root],
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Creates a serialisable checkpoint of the current search state.
    ///
    /// This can be used to pause and resume search work in a separate
    /// process.
    #[must_use]
    pub fn checkpoint(&self) -> GameSearchCheckpoint<E>
    where
        E: serde::Serialize + for<'de> serde::Deserialize<'de>,
        E::Action: serde::Serialize + for<'de> serde::Deserialize<'de> + Clone,
    {
        GameSearchCheckpoint {
            root_env: self.root_env.clone(),
            config: self.config.clone(),
            nodes: self.nodes.clone(),
        }
    }

    /// Restores a search state from a checkpoint.
    #[must_use]
    pub fn restore(checkpoint: GameSearchCheckpoint<E>) -> Self
    where
        E: serde::Serialize + for<'de> serde::Deserialize<'de>,
        E::Action: serde::Serialize + for<'de> serde::Deserialize<'de> + Clone,
    {
        Self {
            root_env: checkpoint.root_env,
            config: checkpoint.config,
            nodes: checkpoint.nodes,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Runs the search for the configured number of iterations.
    ///
    /// Returns the best root action (the one with the most visits),
    /// or `None` if no legal actions exist.
    pub fn run(&mut self) -> Option<E::Action> {
        for _ in 0..self.config.iterations {
            let mut env = self.root_env.clone();

            // 1. Selection — descend using UCT variants.
            let (node_id, mut path) = self.select(&mut env);

            // 2. Expansion — add one untried child if node is expandable.
            let state = env.evaluate();
            if state == GameState::Ongoing && self.should_expand(node_id) {
                let expanded = self.expand(node_id, &mut env);
                if expanded != node_id {
                    path.push(expanded);
                }
            } else if state != GameState::Ongoing {
                self.nodes[node_id as usize].terminal = true;
            }

            // 3. Simulation — random rollout to terminal state.
            let reward = self.simulate(&mut env);

            // 4. Backpropagation — update statistics up to root.
            self.backpropagate(&path, reward);
        }

        self.best_root_action()
    }

    /// Returns statistics for each root child (for diagnostics or visualization).
    pub fn root_stats(&self) -> Vec<(E::Action, NodeStats)>
    where
        E::Action: Clone,
    {
        let root = &self.nodes[0];
        root.children
            .iter()
            .filter_map(|&child_id| {
                let child = &self.nodes[child_id as usize];
                let action = child.action.clone()?;
                let avg = if child.visits > 0 {
                    child.cumulative_reward / f64::from(child.visits)
                } else {
                    0.0
                };
                Some((
                    action,
                    NodeStats {
                        visits: child.visits,
                        average_reward: avg,
                        children_count: child.children.len(),
                        unexpanded_count: child.unexpanded.len(),
                    },
                ))
            })
            .collect()
    }

    /// Returns the total number of nodes in the tree.
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of simulations run so far.
    pub fn total_simulations(&self) -> u32 {
        self.nodes[0].visits
    }

    /// Returns whether the search uses RAVE blending.
    #[must_use]
    pub fn uses_rave(&self) -> bool {
        self.config.rave.enabled
    }
}

mod phases;

#[cfg(test)]
mod tests;
