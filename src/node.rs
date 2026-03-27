//! MCTS tree node with UCT and RAVE statistics.

/// Statistics exposed for diagnostics and visualization.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct NodeStats {
    /// Number of visits to this node.
    pub visits: u32,
    /// Average reward (cumulative / visits).
    pub average_reward: f64,
    /// Number of children this node has.
    pub children_count: usize,
    /// Number of unexpanded actions remaining.
    pub unexpanded_count: usize,
}

/// A node in the MCTS tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Node<A> {
    /// Parent node index (`None` for root).
    pub parent: Option<u32>,
    /// Action that led to this node from parent (`None` for root).
    pub action: Option<A>,
    /// Child node indices.
    pub children: Vec<u32>,
    /// Actions not yet expanded.
    pub unexpanded: Vec<A>,
    /// Visit count.
    pub visits: u32,
    /// Cumulative reward from all simulations through this node.
    pub cumulative_reward: f64,
    /// RAVE cumulative reward (all simulations in subtree).
    pub rave_cumulative: f64,
    /// RAVE visit count.
    pub rave_visits: u32,
    /// Whether this node is a terminal state.
    pub terminal: bool,
}

impl<A: Clone> Node<A> {
    /// Creates a root node seeded with the legal actions available at the root state.
    ///
    /// # Parameters
    ///
    /// - `legal_actions`: Actions that may be expanded from the root.
    ///
    /// # Returns
    ///
    /// Returns a root [`Node`] with empty statistics and no parent/action metadata.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn root(legal_actions: Vec<A>) -> Self {
        Self {
            parent: None,
            action: None,
            children: Vec::new(),
            unexpanded: legal_actions,
            visits: 0,
            cumulative_reward: 0.0,
            rave_cumulative: 0.0,
            rave_visits: 0,
            terminal: false,
        }
    }

    /// Creates a child node linked to an existing parent and action.
    ///
    /// # Parameters
    ///
    /// - `parent`: Parent node index.
    /// - `action`: Action that led from the parent into this node.
    /// - `legal_actions`: Actions available for later expansion from this node.
    ///
    /// # Returns
    ///
    /// Returns a new child [`Node`] with zeroed statistics.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn child(parent: u32, action: A, legal_actions: Vec<A>) -> Self {
        Self {
            parent: Some(parent),
            action: Some(action),
            children: Vec::new(),
            unexpanded: legal_actions,
            visits: 0,
            cumulative_reward: 0.0,
            rave_cumulative: 0.0,
            rave_visits: 0,
            terminal: false,
        }
    }

    /// Reports whether the node has expanded every currently known action.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns `true` when `unexpanded` is empty.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn is_fully_expanded(&self) -> bool {
        self.unexpanded.is_empty()
    }

    /// Applies a standard UCT backpropagation update to the node.
    ///
    /// # Parameters
    ///
    /// - `reward`: Simulation reward to accumulate.
    ///
    /// # Returns
    ///
    /// This function returns no value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn apply_uct_update(&mut self, reward: f64) {
        self.visits += 1;
        self.cumulative_reward += reward;
    }

    /// Applies an AMAF/RAVE update to the node.
    ///
    /// # Parameters
    ///
    /// - `reward`: Reward contribution to accumulate in the RAVE statistics.
    ///
    /// # Returns
    ///
    /// This function returns no value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn apply_rave_update(&mut self, reward: f64) {
        self.rave_visits += 1;
        self.rave_cumulative += reward;
    }

    /// Calculates the node's plain UCT score.
    ///
    /// # Parameters
    ///
    /// - `parent_visits`: Visit count of the parent node.
    /// - `exploration_constant`: Exploration multiplier to apply.
    ///
    /// # Returns
    ///
    /// Returns `f64::INFINITY` for unvisited nodes or the standard exploitation plus
    /// exploration score otherwise.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn uct_score(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        self.uct_score_with_rave(parent_visits, exploration_constant, false, 0.0)
    }

    /// Calculates a blended UCT+RAVE score for this node.
    ///
    /// # Parameters
    ///
    /// - `parent_visits`: Visit count of the parent node.
    /// - `exploration_constant`: Exploration multiplier to apply.
    /// - `rave_enabled`: Whether RAVE blending should be used.
    /// - `rave_bias`: Bias factor controlling how strongly RAVE influences the score.
    ///
    /// # Returns
    ///
    /// Returns a blended score, or the plain UCT score when RAVE is disabled or empty.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn uct_score_with_rave(
        &self,
        parent_visits: u32,
        exploration_constant: f64,
        rave_enabled: bool,
        rave_bias: f64,
    ) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }

        let parent_visits = f64::from(parent_visits);
        let visits = f64::from(self.visits);
        let exploitation = self.cumulative_reward / visits;
        let exploration = exploration_constant * (parent_visits.ln() / visits).sqrt();
        let uct = exploitation + exploration;

        if !rave_enabled || self.rave_visits == 0 {
            return uct;
        }

        let beta = rave_bias / (rave_bias + f64::from(self.rave_visits));
        let rave = self.rave_cumulative / f64::from(self.rave_visits);
        (1.0 - beta) * uct + beta * rave
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_has_no_parent() {
        let node = Node::root(vec![1, 2, 3]);
        assert!(node.parent.is_none());
        assert!(node.action.is_none());
        assert_eq!(node.unexpanded.len(), 3);
    }

    #[test]
    fn child_has_parent_and_action() {
        let node = Node::child(0, 42, vec![1, 2]);
        assert_eq!(node.parent, Some(0));
        assert_eq!(node.action, Some(42));
        assert_eq!(node.unexpanded.len(), 2);
    }

    #[test]
    fn unexpanded_node_not_fully_expanded() {
        let node = Node::<i32>::root(vec![1, 2]);
        assert!(!node.is_fully_expanded());
    }

    #[test]
    fn empty_unexpanded_is_fully_expanded() {
        let node = Node::<i32>::root(vec![]);
        assert!(node.is_fully_expanded());
    }

    #[test]
    fn uct_infinity_for_unvisited() {
        let node = Node::<i32>::root(vec![]);
        assert!(node.uct_score(100, 1.414).is_infinite());
    }

    #[test]
    fn uct_finite_for_visited() {
        let mut node = Node::<i32>::root(vec![]);
        node.visits = 10;
        node.cumulative_reward = 5.0;
        let score = node.uct_score(100, 1.414);
        assert!(score.is_finite());
        assert!(score > 0.0);
    }

    #[test]
    fn applies_updates() {
        let mut node = Node::<i32>::root(vec![]);
        node.apply_uct_update(0.5);
        node.apply_rave_update(1.0);

        assert_eq!(node.visits, 1);
        assert_eq!(node.rave_visits, 1);
        assert!((node.cumulative_reward - 0.5).abs() < f64::EPSILON);
        assert!((node.rave_cumulative - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn uct_with_rave_favors_positive_blend() {
        let mut node = Node::<i32>::root(vec![]);
        node.visits = 10;
        node.cumulative_reward = 4.0;
        node.rave_visits = 5;
        node.rave_cumulative = 20.0;

        let score = node.uct_score_with_rave(25, 1.414, true, 100.0);
        let no_rave = node.uct_score_with_rave(25, 1.414, false, 0.0);

        assert!(score > no_rave);
    }
}
