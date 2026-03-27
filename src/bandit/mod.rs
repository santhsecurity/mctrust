//! Bandit-based MCTS — explore/exploit search for flat action spaces with RAVE.
//!
//! Unlike [`crate::GameSearch`] which builds a game tree, [`BanditSearch`]
//! operates on a pre-enumerated set of "arms" grouped by category.
//! It uses UCT + RAVE to decide which arm to pull next, and you
//! feed back rewards.
//!
//! # Use Cases
//!
//! - **Fuzzing**: arms are test inputs, groups are mutation categories
//! - **Hyperparameter search**: arms are configurations, groups are strategy families
//! - **Security testing**: arms are probes, groups are methods/endpoints
//! - **A/B testing**: arms are variants, groups are features

pub mod config;
mod node;

#[cfg(test)]
mod tests;

#[allow(unused_imports)] // Re-exported for downstream consumers
pub use config::{BanditConfig, BanditConfigBuilder};
use node::BanditNode;
pub use node::GroupStats;

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;
///
/// Arms are grouped by category. The engine maintains a two-level tree:
/// root → group nodes → arm selection within groups.
///
/// # Usage Flow
///
/// 1. Create with [`BanditSearch::new`].
/// 2. Register arms via [`add_arm`](BanditSearch::add_arm).
/// 3. Loop: call [`next_arm`](BanditSearch::next_arm) → evaluate → [`observe`](BanditSearch::observe).
/// 4. Query statistics via [`group_stats`](BanditSearch::group_stats).
pub struct BanditSearch {
    config: BanditConfig,

    /// Flat node arena. Index 0 is root.
    nodes: Vec<BanditNode>,

    /// Group ID → node index mapping.
    group_to_node: HashMap<u32, u32>,

    /// Arm ID → (node index) for backpropagation.
    arm_to_node: HashMap<u64, u32>,

    /// Total pulls executed.
    pulls_executed: u64,

    /// RNG for tie-breaking.
    rng: rand::rngs::StdRng,
}

/// Serializable checkpoint for restoring bandit search progress.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BanditSearchCheckpoint {
    /// Search configuration at checkpoint time.
    pub(crate) config: BanditConfig,
    /// Full two-level tree snapshot.
    nodes: Vec<BanditNode>,
    /// Group mapping.
    group_to_node: HashMap<u32, u32>,
    /// Arm mapping.
    arm_to_node: HashMap<u64, u32>,
    /// Total pulls executed.
    pulls_executed: u64,
}

impl BanditSearch {
    /// Creates a new bandit search engine with an entropy-seeded RNG.
    ///
    /// # Parameters
    ///
    /// - `config`: Search hyperparameters controlling exploration, RAVE, and pull budget.
    ///
    /// # Returns
    ///
    /// Returns an empty [`BanditSearch`] with only the root node initialized.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::{BanditConfig, BanditSearch};
    ///
    /// let search = BanditSearch::new(BanditConfig::default());
    /// assert_eq!(search.total_pulls(), 0);
    /// ```
    #[must_use]
    pub fn new(config: BanditConfig) -> Self {
        Self::with_seed(config, rand::rngs::StdRng::from_entropy())
    }

    /// Creates a new bandit search with a deterministic RNG seed.
    ///
    /// # Parameters
    ///
    /// - `config`: Search hyperparameters.
    /// - `seed`: Seed used to initialize the RNG.
    ///
    /// # Returns
    ///
    /// Returns an empty [`BanditSearch`] with deterministic tie-breaking behavior.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn new_seeded(config: BanditConfig, seed: u64) -> Self {
        Self::with_seed(config, rand::rngs::StdRng::seed_from_u64(seed))
    }

    fn with_seed(config: BanditConfig, rng: rand::rngs::StdRng) -> Self {
        // Create root node.
        let root = BanditNode {
            visits: 0,
            reward: 0.0,
            rave_visits: 0,
            rave_reward: 0.0,
            group_id: u32::MAX,
            bias: 0.0,
            children: Vec::new(),
            arms: Vec::new(),
            next_untried: 0,
        };

        Self {
            config,
            nodes: vec![root],
            group_to_node: HashMap::new(),
            arm_to_node: HashMap::new(),
            pulls_executed: 0,
            rng,
        }
    }

    /// Creates a serializable checkpoint of the current bandit search state.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns a [`BanditSearchCheckpoint`] containing nodes, mappings, and budget state.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn checkpoint(&self) -> BanditSearchCheckpoint {
        BanditSearchCheckpoint {
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            group_to_node: self.group_to_node.clone(),
            arm_to_node: self.arm_to_node.clone(),
            pulls_executed: self.pulls_executed,
        }
    }

    /// Restores a bandit search from a checkpoint using a fresh entropy-seeded RNG.
    ///
    /// # Parameters
    ///
    /// - `checkpoint`: Previously captured search state.
    ///
    /// # Returns
    ///
    /// Returns a [`BanditSearch`] resumed from the checkpoint.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn restore(checkpoint: BanditSearchCheckpoint) -> Self {
        Self {
            config: checkpoint.config,
            nodes: checkpoint.nodes,
            group_to_node: checkpoint.group_to_node,
            arm_to_node: checkpoint.arm_to_node,
            pulls_executed: checkpoint.pulls_executed,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Restores a bandit search from a checkpoint with a deterministic RNG seed.
    ///
    /// # Parameters
    ///
    /// - `checkpoint`: Previously captured search state.
    /// - `seed`: Seed used for deterministic tie-breaking after restore.
    ///
    /// # Returns
    ///
    /// Returns a [`BanditSearch`] resumed from the checkpoint.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn restore_with_seed(checkpoint: BanditSearchCheckpoint, seed: u64) -> Self {
        Self {
            config: checkpoint.config,
            nodes: checkpoint.nodes,
            group_to_node: checkpoint.group_to_node,
            arm_to_node: checkpoint.arm_to_node,
            pulls_executed: checkpoint.pulls_executed,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Registers an arm with the given group.
    ///
    /// Must be called before the first [`next_arm`](Self::next_arm).
    /// Arms within the same group share RAVE statistics.
    /// # Parameters
    ///
    /// - `arm_id`: Unique identifier for the arm to register.
    /// - `group_id`: Group identifier used for RAVE sharing and group selection.
    ///
    /// # Returns
    ///
    /// This function returns no value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::{BanditConfig, BanditSearch};
    ///
    /// let mut search = BanditSearch::new_seeded(BanditConfig::default(), 7);
    /// search.add_arm(10, 1);
    /// assert_eq!(search.next_arm(), Some(10));
    /// ```
    pub fn add_arm(&mut self, arm_id: u64, group_id: u32) {
        if self.arm_to_node.contains_key(&arm_id) {
            return;
        }

        if self.nodes.len() == u32::MAX as usize {
            return;
        }

        // Lazily create the group node.
        let node_idx = if let Some(&idx) = self.group_to_node.get(&group_id) {
            idx
        } else {
            // Safety: conversion is validated below to avoid truncation.
            let Ok(idx) = u32::try_from(self.nodes.len()) else {
                return;
            };
            self.nodes.push(BanditNode {
                visits: 0,
                reward: 0.0,
                rave_visits: 0,
                rave_reward: 0.0,
                group_id,
                bias: 0.0,
                children: Vec::new(),
                arms: Vec::new(),
                next_untried: 0,
            });
            self.nodes[0].children.push(idx);

            self.group_to_node.insert(group_id, idx);
            idx
        };

        self.nodes[node_idx as usize].arms.push(arm_id);
        self.arm_to_node.insert(arm_id, node_idx);
    }

    /// Selects the next arm to pull using UCT + RAVE.
    ///
    /// Returns `None` when the budget is exhausted or all arms have been tried.
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the next arm identifier to evaluate, or `None` when the pull budget is
    /// exhausted or every registered arm has been tried.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn next_arm(&mut self) -> Option<u64> {
        // Budget check.
        if self.config.max_pulls > 0 && self.pulls_executed >= self.config.max_pulls {
            return None;
        }

        // No groups registered.
        let available_groups: Vec<u32> = self
            .nodes
            .first()
            .map(|root| {
                root.children
                    .iter()
                    .copied()
                    .filter(|idx| self.nodes[*idx as usize].has_untried())
                    .collect()
            })
            .unwrap_or_default();
        let mut available_groups = available_groups;
        available_groups.shuffle(&mut self.rng);

        let group_idx = available_groups.iter().copied().max_by(|&a, &b| {
            let sa = self.nodes[a as usize].score(self.nodes[0].visits, &self.config);
            let sb = self.nodes[b as usize].score(self.nodes[0].visits, &self.config);
            let ord = sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal);
            if ord == std::cmp::Ordering::Equal {
                self.nodes[a as usize]
                    .bias
                    .partial_cmp(&self.nodes[b as usize].bias)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                ord
            }
        })?;

        // Expand one untried arm from selected group.
        let node = &mut self.nodes[group_idx as usize];
        if node.next_untried < node.arms.len() {
            let arm_id = node.arms[node.next_untried];
            node.next_untried += 1;
            self.pulls_executed += 1;
            Some(arm_id)
        } else {
            None
        }
    }

    /// Reports the reward for a previously pulled arm.
    ///
    /// Triggers backpropagation and RAVE cross-group updates.
    /// # Parameters
    ///
    /// - `arm_id`: Previously selected arm identifier.
    /// - `reward`: Observed reward to backpropagate.
    ///
    /// # Returns
    ///
    /// This function returns no value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn observe(&mut self, arm_id: u64, reward: f64) {
        let Some(node_idx) = self.arm_to_node.get(&arm_id) else {
            return;
        };

        // Backpropagate: group node → root.
        let node_idx = *node_idx;
        let group_node = &mut self.nodes[node_idx as usize];
        group_node.visits += 1;
        group_node.reward += reward;

        {
            let root = &mut self.nodes[0];
            root.visits += 1;
            root.reward += reward;
        }

        // RAVE: boost sibling groups with decayed reward.
        // Skip entirely when rave_bias is zero (RAVE disabled).
        if self.config.rave_bias > 0.0 {
            let sibling_groups: Vec<u32> = self.nodes[0].children.clone();
            for sibling_idx in sibling_groups {
                if sibling_idx == node_idx {
                    continue;
                }

                let sibling = &mut self.nodes[sibling_idx as usize];
                sibling.rave_visits += 1;
                sibling.rave_reward += reward;
            }
        }
    }

    /// Sets an external bias on a group node.
    ///
    /// Use this to inject domain-specific priors (e.g., from a mixture-of-experts
    /// gating network, from prior scan results, or from static analysis).
    /// # Parameters
    ///
    /// - `group_id`: Target group identifier.
    /// - `bias`: Bias to apply during group tie-breaking.
    ///
    /// # Returns
    ///
    /// This function returns no value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn set_group_bias(&mut self, group_id: u32, bias: f64) {
        if let Some(&node_idx) = self.group_to_node.get(&group_id) {
            self.nodes[node_idx as usize].bias = bias;
        }
    }

    /// Returns a snapshot of per-group search statistics.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns one [`GroupStats`] record for each registered group.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn group_stats(&self) -> Vec<GroupStats> {
        self.nodes[0]
            .children
            .iter()
            .map(|&idx| {
                let node = &self.nodes[idx as usize];
                let avg = if node.visits > 0 {
                    node.reward / f64::from(node.visits)
                } else {
                    0.0
                };
                GroupStats {
                    group_id: node.group_id,
                    visits: node.visits,
                    average_reward: avg,
                    total_arms: node.arms.len(),
                    explored_arms: node.next_untried,
                    rave_visits: node.rave_visits,
                }
            })
            .collect()
    }

    /// Returns the total number of pulls executed so far.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the cumulative pull count.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn total_pulls(&self) -> u64 {
        self.pulls_executed
    }
}
