use mctrust::{Environment, GameSearch, GameState, Heuristic, Reward, SearchConfig};

#[derive(Clone)]
struct SphereMinimizer {
    x: f64,
    step: f64,
}

#[derive(Clone, Debug, PartialEq)]
enum Action {
    Dec,
    Inc,
}

impl SphereMinimizer {
    fn reward_for(&self) -> f64 {
        // Lower absolute value of x is better. Convert to a bounded reward.
        1.0 / (1.0 + self.x.abs())
    }
}

impl Environment for SphereMinimizer {
    type Action = Action;

    fn legal_actions(&self) -> Vec<Action> {
        vec![Action::Dec, Action::Inc]
    }

    fn apply(&mut self, action: &Action) {
        match action {
            Action::Dec => self.x -= self.step,
            Action::Inc => self.x += self.step,
        }
    }

    fn evaluate(&self) -> GameState {
        if self.x.abs() <= 1e-3 {
            GameState::Win(Reward::WIN)
        } else if self.x.abs() >= 12.0 {
            GameState::Loss
        } else {
            GameState::Ongoing
        }
    }

    fn heuristic(&self) -> Heuristic {
        Heuristic::from_reward(Reward::new(self.reward_for()))
    }
}

fn main() {
    let game = SphereMinimizer { x: 4.0, step: 0.5 };
    let config = SearchConfig::builder()
        .iterations(4_000)
        .max_depth(200)
        .tree_policy(mctrust::TreePolicy::Uct)
        .heuristic_weight(0.7)
        .build();

    let mut search = GameSearch::new(game, config);
    if let Some(best) = search.run() {
        println!("best action toward minimum: {best:?}");
    }
}
