//! Reward representation for MCTS outcomes.

use std::ops::{Add, AddAssign};

/// Quantitative reward for an MCTS state or simulation outcome.
///
/// Rewards are typically in `[-1.0, 1.0]` but the engine imposes no range
/// restriction. Higher values indicate better outcomes.
///
/// # Constants
///
/// - [`Reward::WIN`] — maximum positive outcome (`1.0`)
/// - [`Reward::LOSS`] — maximum negative outcome (`-1.0`)
/// - [`Reward::DRAW`] — neutral outcome (`0.0`)
///
/// # Examples
///
/// ```
/// use mctrust::Reward;
///
/// let r = Reward::new(0.75);
/// assert_eq!(r.value(), 0.75);
/// assert_eq!((Reward::WIN + Reward::DRAW).value(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Reward(f64);

impl Reward {
    /// Maximum positive outcome.
    pub const WIN: Self = Self(1.0);

    /// Maximum negative outcome.
    pub const LOSS: Self = Self(-1.0);

    /// Neutral outcome.
    pub const DRAW: Self = Self(0.0);

    /// Creates a reward wrapper from a raw floating-point value.
    ///
    /// # Parameters
    ///
    /// - `value`: Reward magnitude to store.
    ///
    /// # Returns
    ///
    /// Returns a new [`Reward`] containing `value`.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns the underlying floating-point reward value.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the wrapped reward value.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for Reward {
    fn default() -> Self {
        Self::DRAW
    }
}

impl Add for Reward {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl AddAssign for Reward {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl From<f64> for Reward {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_arithmetic() {
        let a = Reward::new(0.5);
        let b = Reward::new(0.3);
        let sum = a + b;
        assert!((sum.value() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn reward_add_assign() {
        let mut r = Reward::DRAW;
        r += Reward::WIN;
        assert!((r.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reward_constants() {
        assert!((Reward::WIN.value() - 1.0).abs() < f64::EPSILON);
        assert!((Reward::LOSS.value() + 1.0).abs() < f64::EPSILON);
        assert!(Reward::DRAW.value().abs() < f64::EPSILON);
    }

    #[test]
    fn reward_default() {
        assert_eq!(Reward::default(), Reward::DRAW);
    }

    #[test]
    fn reward_from_f64() {
        let r: Reward = 0.42.into();
        assert!((r.value() - 0.42).abs() < f64::EPSILON);
    }
}
