//! # santh-mcts — Production-grade Monte Carlo Tree Search
//!
//! A generic, zero-domain-assumption MCTS implementation with UCT selection,
//! RAVE (Rapid Action Value Estimation), and pluggable environments.
//!
//! ## Design
//!
//! This crate provides two complementary MCTS interfaces:
//!
//! 1. **Environment-based** ([`GameSearch`]) — Classic game-tree MCTS. You define
//!    an [`Environment`] with actions, state transitions, and terminal conditions.
//!    The engine handles selection, expansion, simulation, and backpropagation.
//!    Best for: game AI, optimization, planning problems.
//!
//! 2. **Bandit-based** ([`BanditSearch`]) — Two-level MCTS for explore/exploit
//!    problems where you have a flat set of arms grouped by category.
//!    The engine uses UCT+RAVE to decide which arm to pull next, and you
//!    feed back rewards. Best for: fuzzing, testing, hyperparameter search,
//!    any problem with a pre-enumerated action space.
//!
//! ## Quick Start (Environment-based)
//!
//! ```rust
//! use mctrust::{Environment, GameState, GameSearch, SearchConfig, Reward};
//!
//! #[derive(Clone)]
//! struct MyGame { /* ... */ }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! enum Move { Left, Right }
//!
//! impl Environment for MyGame {
//!     type Action = Move;
//!
//!     fn legal_actions(&self) -> Vec<Move> {
//!         vec![Move::Left, Move::Right]
//!     }
//!
//!     fn apply(&mut self, action: &Move) {
//!         // mutate state
//!     }
//!
//!     fn evaluate(&self) -> GameState {
//!         GameState::Ongoing
//!     }
//! }
//!
//! let game = MyGame { /* ... */ };
//! let config = SearchConfig::default();
//! let mut search = GameSearch::new(game, config);
//! let best_move = search.run();
//! ```
//!
//! ## Quick Start (Bandit-based)
//!
//! ```rust
//! use mctrust::{BanditSearch, BanditConfig};
//!
//! let mut search = BanditSearch::new(BanditConfig::default());
//!
//! // Register 100 arms across 5 groups.
//! for group in 0..5u64 {
//!     for arm in 0..20u64 {
//!         search.add_arm(group * 20 + arm, group as u32);
//!     }
//! }
//!
//! // Pull arms and feed rewards.
//! while let Some(arm_id) = search.next_arm() {
//!     let reward = 0.5; // your evaluation
//!     search.observe(arm_id, reward);
//! }
//! ```
//!
//! ## Features
//!
//! - **UCT selection** with configurable exploration constant
//! - **RAVE** for rapid cross-branch value estimation
//! - **Flat arena storage** for cache-efficient tree traversal
//! - **Budget control** (iteration cap, time budget hooks)
//! - **Configurable reward** — bring your own reward function
//! - **Thread-safe RNG** via `StdRng` (no `ThreadRng` footgun)
//! - **Zero domain assumptions** — works for games, fuzzing, optimization, anything
//!
//! ## References
//!
//! - Kocsis & Szepesvári, "Bandit-based Monte Carlo Planning" (2006)
//! - Gelly & Silver, "Monte-Carlo tree search and rapid action value estimation
//!   in computer Go" (2011)
//! - Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)

#![warn(clippy::pedantic)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]

mod bandit;
mod config;
mod environment;
mod game_search;
mod node;
mod reward;

#[cfg(test)]
mod adversarial_tests;

pub use bandit::{BanditConfig, BanditSearch, BanditSearchCheckpoint};
pub use config::{ProgressiveWideningConfig, SearchConfig, TreePolicy};
pub use environment::Heuristic;
pub use environment::{Environment, GameState};
pub use game_search::{GameSearch, GameSearchCheckpoint};
pub use node::NodeStats;
pub use reward::Reward;
