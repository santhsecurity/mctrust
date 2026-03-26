use rand::seq::SliceRandom;
use rand::Rng;
use crate::config::{ProgressiveWideningConfig, TreePolicy};
use crate::environment::{Environment, GameState};
use crate::node::Node;
use crate::reward::Reward;
use super::GameSearch;

impl<E: Environment> GameSearch<E> {
    // ── Internal phases ─────────────────────────────────────────

    pub(crate) fn pick_best_child(&mut self, current: u32, env: &mut E) -> Option<u32> {
        let node = &self.nodes[current as usize];
        let parent_visits = node.visits;
        let legal = env.legal_actions();
        let priors = match self.config.tree_policy {
            TreePolicy::Puct { .. } => env.action_priors(&legal),
            _ => None,
        };

        let child_ids: Vec<u32> = node.children.clone();
        let mut best_child = None;
        let mut best_score = f64::NEG_INFINITY;

        for child in child_ids {
            let score = self.selection_score(child, parent_visits, &legal, priors.as_ref());

            if score
                .partial_cmp(&best_score)
                .is_some_and(std::cmp::Ordering::is_gt)
            {
                best_score = score;
                best_child = Some(child);
            }
        }
        best_child
    }

    /// Selection: descend the tree via configured policy, applying actions to the environment clone.
    pub(crate) fn select(&mut self, env: &mut E) -> (u32, Vec<u32>) {
        let mut current = 0u32;
        let mut path = vec![current];

        loop {
            let node = &self.nodes[current as usize];

            // Expand immediately if this is a terminal node, a leaf, or we are
            // still below progressive widening capacity.
            if node.terminal || node.children.is_empty() || self.should_expand(current) {
                return (current, path);
            }

            match self.pick_best_child(current, env) {
                Some(child) => {
                    if let Some(ref action) = self.nodes[child as usize].action {
                        env.apply(action);
                    }
                    path.push(child);
                    current = child;
                }
                // All children are empty — treat as leaf.
                None => return (current, path),
            }
        }
    }

    pub(crate) fn selection_score(
        &mut self,
        child: u32,
        parent_visits: u32,
        legal_actions: &[E::Action],
        priors: Option<&Vec<f64>>,
    ) -> f64 {
        let node = &self.nodes[child as usize];
        let score = node.uct_score_with_rave(
            parent_visits,
            self.config.exploration_constant,
            self.config.rave.enabled,
            self.config.rave.bias,
        );

        match self.config.tree_policy {
            TreePolicy::Puct { prior_weight } => {
                let prior = if let Some(priors) = priors {
                    if priors.is_empty() {
                        1.0
                    } else {
                        let idx = node
                            .action
                            .as_ref()
                            .and_then(|action| legal_actions.iter().position(|a| a == action));
                        if let Some(i) = idx {
                            priors.get(i).copied().unwrap_or(1.0).max(0.0)
                        } else {
                            1.0
                        }
                    }
                } else {
                    1.0
                };

                score + prior * prior_weight
            }
            TreePolicy::ThompsonSampling { temperature } => {
                let noise = (self.rng.gen_range(0.0..1.0) - 0.5) * 2.0 * temperature.max(0.0);
                score + noise
            }
            TreePolicy::Uct => score,
        }
    }

    pub(crate) fn should_expand(&self, node_id: u32) -> bool {
        let node = &self.nodes[node_id as usize];

        if node.is_fully_expanded() {
            return false;
        }

        if let Some(cfg) = &self.config.progressive_widening {
            let max_children = Self::progressive_limit(node.visits, cfg);
            node.children.len() < max_children
        } else {
            true
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub(crate) fn progressive_limit(parent_visits: u32, cfg: &ProgressiveWideningConfig) -> usize {
        let budget = cfg.coefficient * (f64::from(parent_visits)).powf(cfg.exponent);
        let budget = if budget.is_finite() && budget > 0.0 {
            budget.floor().min(usize::MAX as f64) as usize
        } else {
            0
        };
        let candidate = cfg.minimum_children.max(budget);
        candidate.max(cfg.minimum_children)
    }

    /// Expansion: pop one unexpanded action, create a child node.
    pub(crate) fn expand(&mut self, parent_id: u32, env: &mut E) -> u32 {
        let Some(action) = self.nodes[parent_id as usize].unexpanded.pop() else {
            return parent_id;
        };

        env.apply(&action);

        let legal = env.legal_actions();
        let Ok(child_id) = u32::try_from(self.nodes.len()) else {
            return parent_id;
        };
        let mut child = Node::child(parent_id, action, legal);

        let state = env.evaluate();
        if state != GameState::Ongoing {
            child.terminal = true;
        }

        self.nodes.push(child);
        self.nodes[parent_id as usize].children.push(child_id);

        child_id
    }

    /// Simulation: random rollout from the current environment state.
    pub(crate) fn simulate(&mut self, env: &mut E) -> Reward {
        let mut depth = 0usize;
        let depth_limit = env.max_depth().unwrap_or(self.config.max_depth);

        loop {
            match env.evaluate() {
                GameState::Win(r) | GameState::Terminal(r) => return r,
                GameState::Loss => return Reward::LOSS,
                GameState::Draw => return Reward::DRAW,
                GameState::Ongoing => {
                    if depth >= depth_limit {
                        if let Some(heuristic) = env.heuristic().value {
                            return Reward::new(heuristic.value() * self.config.heuristic_weight);
                        }
                        return Reward::DRAW;
                    }

                    let actions = env.legal_actions();
                    if actions.is_empty() {
                        return Reward::DRAW;
                    }

                    // Choose a random action.
                    let action = match actions.choose(&mut self.rng) {
                        Some(a) => a.clone(),
                        // `actions` is non-empty — this branch is unreachable.
                        // Returning DRAW is fail-safe.
                        None => return Reward::DRAW,
                    };

                    env.apply(&action);
                    depth += 1;
                }
            }
        }
    }

    /// Backpropagation: push reward up from root to the selected path.
    pub(crate) fn backpropagate(&mut self, path: &[u32], reward: Reward) {
        let value = reward.value();
        for &node_id in path {
            let node = &mut self.nodes[node_id as usize];
            node.apply_uct_update(value);
            if self.config.rave.enabled {
                node.apply_rave_update(value);
            }
        }
    }

    /// Returns the root child with the most visits (robust child policy).
    pub(crate) fn best_root_action(&self) -> Option<E::Action> {
        let root = &self.nodes[0];
        if root.children.is_empty() {
            return None;
        }

        root.children
            .iter()
            .copied()
            .max_by_key(|&id| self.nodes[id as usize].visits)
            .and_then(|id| self.nodes[id as usize].action.clone())
    }
}


