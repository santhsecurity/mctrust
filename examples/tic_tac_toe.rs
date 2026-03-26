//! MCTS playing Tic-Tac-Toe
//!
//! Run: cargo run --example tic_tac_toe

use mctrust::{Environment, GameSearch, GameState, Reward, SearchConfig};

#[derive(Clone)]
struct TicTacToe {
    board: [Option<bool>; 9],
    x_turn: bool,
}

impl TicTacToe {
    fn new() -> Self {
        Self {
            board: [None; 9],
            x_turn: true,
        }
    }

    fn winner(&self) -> Option<bool> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for line in &LINES {
            match (
                self.board[line[0]],
                self.board[line[1]],
                self.board[line[2]],
            ) {
                (Some(a), Some(b), Some(c)) if a == b && b == c => return Some(a),
                _ => {}
            }
        }

        None
    }

    fn board_to_string(&self) -> String {
        let mut out = String::new();
        for row in 0..3 {
            for col in 0..3 {
                let cell = match self.board[row * 3 + col] {
                    Some(true) => 'X',
                    Some(false) => 'O',
                    None => '.',
                };
                out.push(cell);
                if col < 2 {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        out.push('\n');
        out
    }
}

impl Environment for TicTacToe {
    type Action = usize;

    fn legal_actions(&self) -> Vec<usize> {
        (0..9).filter(|&i| self.board[i].is_none()).collect()
    }

    fn apply(&mut self, action: &usize) {
        self.board[*action] = Some(self.x_turn);
        self.x_turn = !self.x_turn;
    }

    fn evaluate(&self) -> GameState {
        match self.winner() {
            Some(_) => GameState::Win(Reward::WIN),
            None if self.board.iter().all(|c| c.is_some()) => GameState::Draw,
            None => GameState::Ongoing,
        }
    }
}

fn main() {
    let config = SearchConfig::builder().iterations(5_000).build();
    let mut game = TicTacToe::new();

    println!("MCTS Tic-Tac-Toe");
    while matches!(game.evaluate(), GameState::Ongoing) {
        let mut search = GameSearch::new(game.clone(), config.clone());
        let action = search.run();

        match action {
            Some(action) => {
                println!("{} plays {}", if game.x_turn { 'X' } else { 'O' }, action);
                game.apply(&action);
                print!("{}", game.board_to_string());
            }
            None => {
                eprintln!("No legal moves available");
                break;
            }
        }
    }

    println!("Result: {:?}", game.winner());
}
