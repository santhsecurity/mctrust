# mctrust

Monte Carlo Tree Search. UCT, RAVE, progressive widening, configurable policies, optional parallelism. Implement the `Environment` trait for your domain and the search handles the rest.

```rust
use mctrust::{Environment, GameState, SearchConfig, GameSearch, Reward};

// Define your domain
impl Environment for MyGame {
    type Action = usize;
    fn legal_actions(&self) -> Vec<usize> { /* ... */ }
    fn apply(&mut self, action: &usize) { /* ... */ }
    fn evaluate(&self) -> GameState { /* ... */ }
}

// Search
let config = SearchConfig::builder().iterations(10_000).build();
let mut search = GameSearch::new(game, config);
let best_move = search.run();
```

## Not just for games

MCTS works anywhere you have a state space to explore: hyperparameter tuning, robot path planning, scan strategy optimization, molecular design, theorem proving. The `Environment` trait is generic.

## TOML configuration

```toml
iterations = 50000
exploration_constant = 1.41
max_depth = 100
```

## Contributing

Pull requests are welcome. There is no such thing as a perfect crate. If you find a bug, a better API, or just a rough edge, open a PR. We review quickly.

## License

MIT. Copyright 2026 CORUM COLLECTIVE LLC.
