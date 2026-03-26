use super::*;
use crate::{ProgressiveWideningConfig, Reward, TreePolicy};

// ── Test environment: find the target number ────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct NumberGame {
    value: i32,
    target: i32,
    move_count: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
enum Move {
    Inc,
    Dec,
}

impl Environment for NumberGame {
    type Action = Move;

    fn legal_actions(&self) -> Vec<Move> {
        vec![Move::Inc, Move::Dec]
    }

    fn apply(&mut self, action: &Move) {
        match action {
            Move::Inc => {
                self.value += 1;
                self.move_count += 1;
            }
            Move::Dec => {
                self.value -= 1;
                self.move_count += 1;
            }
        }
    }

    fn evaluate(&self) -> GameState {
        if self.value == self.target {
            GameState::Win(Reward::WIN)
        } else if (self.value - self.target).abs() > 20 {
            GameState::Loss
        } else {
            GameState::Ongoing
        }
    }

    fn heuristic(&self) -> crate::environment::Heuristic {
        let distance = f64::from((self.value - self.target).abs());
        crate::environment::Heuristic::from_reward(Reward::new((-distance).powf(0.25)))
    }

    fn max_depth(&self) -> Option<usize> {
        Some(20)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PriorGame {
    state: i32,
}

impl Environment for PriorGame {
    type Action = i32;

    fn legal_actions(&self) -> Vec<i32> {
        if self.state == 0 {
            vec![1, -1]
        } else {
            vec![0]
        }
    }

    fn apply(&mut self, action: &i32) {
        self.state += action;
    }

    fn evaluate(&self) -> GameState {
        if self.state == 2 {
            GameState::Win(Reward::WIN)
        } else if self.state == -2 {
            GameState::Loss
        } else {
            GameState::Ongoing
        }
    }

    fn action_priors(&self, actions: &[Self::Action]) -> Option<Vec<f64>> {
        Some(actions.iter().map(|a| if *a == 1 { 1.0 } else { 0.1 }).collect())
    }
}

#[test]
fn finds_correct_direction() {
    let game = NumberGame {
        value: 0,
        target: 3,
        move_count: 0,
    };
    let config = SearchConfig::builder()
        .iterations(2_000)
        .max_depth(10)
        .build();

    let mut search = GameSearch::with_seed(game, config, 42);
    let best = search.run();

    // Target is +3, so the engine should prefer Inc.
    assert_eq!(best, Some(Move::Inc));
}

#[test]
fn finds_negative_direction() {
    let game = NumberGame {
        value: 0,
        target: -3,
        move_count: 0,
    };
    let config = SearchConfig::builder()
        .iterations(2_000)
        .max_depth(10)
        .build();

    let mut search = GameSearch::with_seed(game, config, 42);
    let best = search.run();

    assert_eq!(best, Some(Move::Dec));
}

#[test]
fn no_actions_returns_none() {
    #[derive(Clone)]
    struct DeadEnd;

    #[derive(Clone, Debug, PartialEq)]
    struct Noop;

    impl Environment for DeadEnd {
        type Action = Noop;

        fn legal_actions(&self) -> Vec<Noop> {
            vec![]
        }

        fn apply(&mut self, _: &Noop) {}

        fn evaluate(&self) -> GameState {
            GameState::Ongoing
        }
    }

    let mut search = GameSearch::new(DeadEnd, SearchConfig::default());
    assert!(search.run().is_none());
}

#[test]
fn root_stats_populated() {
    let game = NumberGame {
        value: 0,
        target: 3,
        move_count: 0,
    };
    let config = SearchConfig::builder().iterations(500).build();
    let mut search = GameSearch::with_seed(game, config, 99);
    search.run();

    let stats = search.root_stats();
    assert_eq!(stats.len(), 2); // Inc and Dec
    let total_visits: u32 = stats.iter().map(|(_, s)| s.visits).sum();
    assert!(total_visits > 0);
    assert!(total_visits <= 500);
}

#[test]
fn tree_grows_with_iterations() {
    let game = NumberGame {
        value: 0,
        target: 5,
        move_count: 0,
    };
    let config = SearchConfig::builder().iterations(100).max_depth(8).build();
    let mut search = GameSearch::with_seed(game, config, 7);
    search.run();

    // Tree should have grown beyond root + 2 children.
    assert!(search.tree_size() > 3);
}

#[test]
fn deterministic_with_same_seed() {
    let game = NumberGame {
        value: 0,
        target: 3,
        move_count: 0,
    };
    let config = SearchConfig::builder().iterations(1_000).max_depth(10).build();

    let mut s1 = GameSearch::with_seed(game.clone(), config.clone(), 42);
    let mut s2 = GameSearch::with_seed(game, config, 42);

    assert_eq!(s1.run(), s2.run());
}

#[test]
fn total_simulations_matches_iterations() {
    let game = NumberGame {
        value: 0,
        target: 3,
        move_count: 0,
    };
    let config = SearchConfig::builder().iterations(200).build();
    let mut search = GameSearch::with_seed(game, config, 1);
    search.run();

    assert_eq!(search.total_simulations(), 200);
}

#[test]
fn puct_policy_uses_priors() {
    let game = PriorGame { state: 0 };
    let config = SearchConfig::builder()
        .iterations(100)
        .tree_policy(TreePolicy::Puct { prior_weight: 2.0 })
        .build();

    let mut search = GameSearch::with_seed(game, config, 12);
    search.run();

    let best = search.best_root_action();
    assert_eq!(best, Some(1));
}

#[test]
fn thompson_policy_runs() {
    let game = PriorGame { state: 0 };
    let config = SearchConfig::builder()
        .iterations(50)
        .tree_policy(TreePolicy::ThompsonSampling { temperature: 0.5 })
        .build();

    let mut search = GameSearch::with_seed(game, config, 33);
    let best = search.run();
    assert_eq!(best, Some(1));
    assert!(search.total_simulations() > 0);
}

#[test]
fn uses_progressive_widening_limit() {
    let game = PriorGame { state: 0 };
    let config = SearchConfig::builder()
        .iterations(20)
        .progressive_widening(ProgressiveWideningConfig {
            minimum_children: 1,
            coefficient: 0.0,
            exponent: 1.0,
        })
        .build();

    let mut search = GameSearch::with_seed(game, config, 5);
    search.run();
    // In this config root can only track one expanded child under strict limit.
    assert_eq!(search.nodes[0].children.len(), 1);
}

#[test]
fn checkpoint_restores_progress() {
    let game = NumberGame {
        value: 0,
        target: 4,
        move_count: 0,
    };
    let config = SearchConfig::builder().iterations(20).build();
    let mut search = GameSearch::with_seed(game, config, 11);
    search.run();
    let checkpoint = search.checkpoint();
    let resumed = GameSearch::restore(checkpoint);
    assert_eq!(search.tree_size(), resumed.tree_size());
}

#[test]
fn uses_uct_rave_toggle() {
    let game = NumberGame {
        value: 0,
        target: 3,
        move_count: 0,
    };
    let config = SearchConfig::builder()
        .iterations(50)
        .rave(crate::config::RaveConfig {
            enabled: false,
            bias: 1.0,
        })
        .build();
    let search = GameSearch::with_seed(game, config, 1);
    assert!(!search.uses_rave());
}

#[test]
fn checkpoint_roundtrip() {
    let game = PriorGame { state: 0 };
    let config = SearchConfig::builder().iterations(80).build();
    let mut search = GameSearch::with_seed(game, config, 1);
    search.run();

    let checkpoint = search.checkpoint();
    let resumed: GameSearch<PriorGame> = GameSearch::restore(checkpoint);
    assert!(resumed.total_simulations() > 0);
}
