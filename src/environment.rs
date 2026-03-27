//! The `Environment` trait — the core integration point for MCTS.

use crate::reward::Reward;

/// Terminal or non-terminal state of an environment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum GameState {
    /// Search can continue from this state.
    Ongoing,
    /// A generalized terminal outcome with an explicit reward.
    Terminal(Reward),
    /// Compatibility helper for positive terminal outcomes.
    Win(Reward),
    /// Compatibility helper for negative terminal outcomes.
    Loss,
    /// Compatibility helper for neutral terminal outcomes.
    Draw,
}

impl GameState {
    /// Returns `true` when the state is terminal.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Ongoing)
    }

    /// Returns the terminal reward when available.
    #[must_use]
    pub fn reward(self) -> Option<Reward> {
        match self {
            Self::Ongoing => None,
            Self::Terminal(reward) | Self::Win(reward) => Some(reward),
            Self::Loss => Some(Reward::LOSS),
            Self::Draw => Some(Reward::DRAW),
        }
    }
}

impl std::fmt::Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ongoing => f.write_str("ongoing"),
            Self::Terminal(reward) => write!(f, "terminal({reward})"),
            Self::Win(reward) => write!(f, "win({reward})"),
            Self::Loss => f.write_str("loss"),
            Self::Draw => f.write_str("draw"),
        }
    }
}

/// Optional heuristic signals for a state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Heuristic {
    /// Estimated value of the current state.
    pub value: Option<Reward>,
}

impl Heuristic {
    /// Creates a heuristic from a reward estimate.
    #[must_use]
    pub fn from_reward(value: Reward) -> Self {
        Self { value: Some(value) }
    }
}

/// A domain environment that the MCTS engine can explore.
///
/// The trait is intentionally broad enough for adversarial games, planning,
/// optimization, scheduling, and scientific search problems.
pub trait Environment: Clone + Send + Sync {
    /// The action type chosen by the search.
    type Action: Clone + Send + Sync + std::fmt::Debug + PartialEq;

    /// Returns all legal actions from the current state.
    fn legal_actions(&self) -> Vec<Self::Action>;

    /// Applies an action to the current state.
    fn apply(&mut self, action: &Self::Action);

    /// Evaluates the current state.
    fn evaluate(&self) -> GameState;

    /// Optional value estimate for non-terminal states.
    ///
    /// This is used when rollouts hit a depth cap or when a domain wants to
    /// inject prior knowledge into the search.
    fn heuristic(&self) -> Heuristic {
        Heuristic::default()
    }

    /// Optional state-specific depth budget override.
    fn max_depth(&self) -> Option<usize> {
        None
    }

    /// Optional action priors for PUCT-style search.
    ///
    /// The returned vector must have the same length and ordering as `actions`.
    /// Invalid priors are ignored and the engine falls back to a uniform prior.
    fn action_priors(&self, _actions: &[Self::Action]) -> Option<Vec<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{GameState, Heuristic, Reward};

    #[test]
    fn game_state_terminal_detection() {
        assert!(GameState::Terminal(Reward::WIN).is_terminal());
        assert!(!GameState::Ongoing.is_terminal());
        assert_eq!(GameState::Loss.reward(), Some(Reward::LOSS));
    }

    #[test]
    fn heuristic_default_and_constructor() {
        assert_eq!(Heuristic::default(), Heuristic { value: None });
        let h = Heuristic::from_reward(Reward::new(0.25));
        assert_eq!(h.value, Some(Reward::new(0.25)));
    }

    #[test]
    fn format_terminal_states() {
        assert_eq!(format!("{}", GameState::Win(Reward::WIN)), "win(1)");
        assert_eq!(format!("{}", GameState::Loss), "loss");
        assert_eq!(format!("{}", GameState::Draw), "draw");
    }
}
