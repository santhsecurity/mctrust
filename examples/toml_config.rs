use std::error::Error;
use std::path::Path;

use mctrust::{Environment, GameSearch, GameState, Heuristic, Reward, SearchConfig, TreePolicy};

#[derive(Clone)]
struct ConfigToy {
    value: i32,
}

impl Environment for ConfigToy {
    type Action = i32;

    fn legal_actions(&self) -> Vec<i32> {
        if self.value == 0 {
            vec![1, -1]
        } else {
            vec![]
        }
    }

    fn apply(&mut self, action: &i32) {
        self.value += action;
    }

    fn evaluate(&self) -> GameState {
        if self.value == 0 {
            GameState::Win(Reward::WIN)
        } else {
            GameState::Draw
        }
    }

    fn heuristic(&self) -> Heuristic {
        Heuristic::from_reward(if self.value == 0 {
            Reward::WIN
        } else {
            Reward::DRAW
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config_text = r#"
iterations = 512
max_depth = 8
heuristic_weight = 0.7

[tree_policy]
kind = "puct"
prior_weight = 0.9

[rave]
enabled = true
bias = 250.0
"#;

    let cfg_path = std::env::temp_dir().join("mctrust_toml_example.toml");
    std::fs::write(&cfg_path, config_text)?;

    let config = SearchConfig::from_toml_file(Path::new(&cfg_path))?;
    assert!(matches!(config.tree_policy, TreePolicy::Puct { .. }));

    let mut search = GameSearch::new(ConfigToy { value: 4 }, config);
    let best = search.run();
    println!("best action from TOML config: {best:?}");

    Ok(())
}
