use super::config::BanditConfig;
use serde::{Deserialize, Serialize};

/// Internal node for the two-level bandit tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BanditNode {
    /// Visit count.
    pub visits: u32,

    /// Cumulative reward.
    pub reward: f64,

    /// RAVE visit count (from sibling group observations).
    pub rave_visits: u32,

    /// RAVE cumulative reward.
    pub rave_reward: f64,

    /// Group this node belongs to.
    pub group_id: u32,

    /// External bias (set by caller for domain-specific prioritization).
    pub bias: f64,

    /// Child node indices.
    pub children: Vec<u32>,

    /// Arm IDs belonging to this node (leaf level).
    pub arms: Vec<u64>,

    /// Index of next untried arm.
    pub next_untried: usize,
}

impl BanditNode {
    /// UCT + RAVE + bias score.
    pub(crate) fn score(&self, parent_visits: u32, config: &BanditConfig) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }

        let exploitation = self.reward / f64::from(self.visits);
        let exploration = config.exploration_constant
            * (f64::from(parent_visits).ln() / f64::from(self.visits)).sqrt();

        // RAVE blending: standard equivalence parameter formula (Gelly & Silver 2007).
        // beta = rave_visits / (rave_visits + visits + 4 * b^2 * visits * rave_visits)
        // When rave_visits >> visits, beta → 1 (trust RAVE).
        // When visits >> rave_visits, beta → 0 (trust UCT).
        let beta = if self.rave_visits > 0 {
            let n = f64::from(self.visits);
            let nr = f64::from(self.rave_visits);
            let b2 = config.rave_bias * config.rave_bias;
            nr / (nr + n + 4.0 * b2 * n * nr)
        } else {
            0.0
        };

        let rave_value = if self.rave_visits > 0 {
            self.rave_reward / f64::from(self.rave_visits)
        } else {
            0.0
        };

        let uct = exploitation + exploration;
        (1.0 - beta) * uct + beta * rave_value + self.bias
    }

    /// Returns `true` if there are untried arms remaining.
    pub(crate) fn has_untried(&self) -> bool {
        self.next_untried < self.arms.len()
    }
}

/// Read-only statistics for a group in [`crate::BanditSearch`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupStats {
    /// The group identifier.
    pub group_id: u32,
    /// Number of direct visits (arms pulled from this group).
    pub visits: u32,
    /// Average reward across all visits.
    pub average_reward: f64,
    /// Total number of registered arms.
    pub total_arms: usize,
    /// Number of arms that have been pulled at least once.
    pub explored_arms: usize,
    /// Number of RAVE updates from sibling groups.
    pub rave_visits: u32,
}
