use mctrust::{Environment, GameSearch, GameState, Reward, SearchConfig};

#[derive(Clone)]
struct CountingGame {
    value: i32,
    target: i32,
}

#[derive(Clone, Debug, PartialEq)]
enum Move {
    Increment,
    Decrement,
}

impl Environment for CountingGame {
    type Action = Move;

    fn legal_actions(&self) -> Vec<Move> {
        vec![Move::Increment, Move::Decrement]
    }

    fn apply(&mut self, action: &Move) {
        match action {
            Move::Increment => self.value += 1,
            Move::Decrement => self.value -= 1,
        }
    }

    fn evaluate(&self) -> GameState {
        if self.value == self.target {
            GameState::Win(Reward::WIN)
        } else {
            GameState::Ongoing
        }
    }
}

fn main() {
    let game = CountingGame {
        value: 0,
        target: 5,
    };
    let config = SearchConfig::builder().iterations(1_000).build();
    let mut search = GameSearch::new(game, config);

    let best_move = search.run();
    println!("Best move: {:?}", best_move);
}
