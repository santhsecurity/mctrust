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
    /// Creates a new game-tree search from an environment and configuration.
    ///
    /// # Parameters
    ///
    /// - `environment`: Root state to search from.
    /// - `config`: Search hyperparameters.
    ///
    /// # Returns
    ///
    /// Returns a [`GameSearch`] with a root node populated from the environment's legal
    /// actions.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::{Environment, GameSearch, GameState, SearchConfig};
    ///
    /// #[derive(Clone)]
    /// struct Env;
    ///
    /// impl Environment for Env {
    ///     type Action = ();
    ///     fn legal_actions(&self) -> Vec<Self::Action> { vec![()] }
    ///     fn apply(&mut self, _action: &Self::Action) {}
    ///     fn evaluate(&self) -> GameState { GameState::Draw }
    /// }
    ///
    /// let search = GameSearch::new(Env, SearchConfig::default());
    /// assert_eq!(search.tree_size(), 1);
    /// ```
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

    /// Creates a new game-tree search with a deterministic RNG seed.
    ///
    /// # Parameters
    ///
    /// - `environment`: Root state to search from.
    /// - `config`: Search hyperparameters.
    /// - `seed`: Seed used for rollout randomness and tie-breaking.
    ///
    /// # Returns
    ///
    /// Returns a deterministic [`GameSearch`].
    ///
    /// # Panics
    ///
    /// This function does not panic.
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
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns a [`GameSearchCheckpoint`] containing the root environment, config, and tree.
    ///
    /// # Panics
    ///
    /// This function does not panic.
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
    ///
    /// # Parameters
    ///
    /// - `checkpoint`: Previously captured search state.
    ///
    /// # Returns
    ///
    /// Returns a [`GameSearch`] resumed from the checkpoint.
    ///
    /// # Panics
    ///
    /// This function does not panic.
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
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the most-visited root action, or `None` if the environment has no legal actions.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::{Environment, GameSearch, GameState, Reward, SearchConfig};
    ///
    /// #[derive(Clone)]
    /// struct Env(bool);
    ///
    /// #[derive(Clone, Debug, PartialEq)]
    /// enum Action { Win }
    ///
    /// impl Environment for Env {
    ///     type Action = Action;
    ///     fn legal_actions(&self) -> Vec<Self::Action> { if self.0 { vec![] } else { vec![Action::Win] } }
    ///     fn apply(&mut self, _action: &Self::Action) { self.0 = true; }
    ///     fn evaluate(&self) -> GameState { if self.0 { GameState::Win(Reward::WIN) } else { GameState::Ongoing } }
    /// }
    ///
    /// let mut search = GameSearch::with_seed(Env(false), SearchConfig::builder().iterations(8).build(), 1);
    /// assert_eq!(search.run(), Some(Action::Win));
    /// ```
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

    /// Returns statistics for each root child.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns `(action, stats)` pairs for every expanded child of the root.
    ///
    /// # Panics
    ///
    /// This function does not panic.
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

    /// Returns the total number of nodes currently stored in the search tree.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the arena length.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of simulations run so far.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the root visit count.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn total_simulations(&self) -> u32 {
        self.nodes[0].visits
    }

    /// Reports whether the configured search uses RAVE blending.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns `true` when `config.rave.enabled` is set.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn uses_rave(&self) -> bool {
        self.config.rave.enabled
    }
}

mod phases;

#[cfg(test)]
mod tests;
