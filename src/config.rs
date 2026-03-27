//! Search configuration and policy types.

use std::path::Path;

/// Tree policy used during child selection.
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TreePolicy {
    /// Classic Upper Confidence Bounds for Trees.
    #[default]
    Uct,
    /// AlphaZero-style PUCT using action priors.
    Puct {
        /// Prior contribution multiplier.
        prior_weight: f64,
    },
    /// Thompson-style optimistic sampling from a node's reward estimate.
    ThompsonSampling {
        /// Standard deviation multiplier used to perturb the sampled value.
        temperature: f64,
    },
}

/// Configuration for AMAF/RAVE value blending.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RaveConfig {
    /// Enables or disables RAVE.
    pub enabled: bool,
    /// Weight used for visit-count-based RAVE decay.
    pub bias: f64,
}

impl Default for RaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bias: 300.0,
        }
    }
}

/// Progressive widening for large or continuous action spaces.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProgressiveWideningConfig {
    /// Base number of children always allowed.
    pub minimum_children: usize,
    /// The widening coefficient `k`.
    pub coefficient: f64,
    /// The exponent `alpha` in `k * visits^alpha`.
    pub exponent: f64,
}

impl Default for ProgressiveWideningConfig {
    fn default() -> Self {
        Self {
            minimum_children: 1,
            coefficient: 1.5,
            exponent: 0.5,
        }
    }
}

/// Configuration for [`crate::GameSearch`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Number of simulations to execute.
    pub iterations: usize,
    /// Exploration constant used by UCT and as a fallback with other policies.
    pub exploration_constant: f64,
    /// Default rollout depth cap.
    pub max_depth: usize,
    /// Tree policy used for child selection.
    pub tree_policy: TreePolicy,
    /// Heuristic blend weight when a heuristic estimate is available.
    pub heuristic_weight: f64,
    /// Enables and configures RAVE.
    pub rave: RaveConfig,
    /// Optional progressive widening.
    pub progressive_widening: Option<ProgressiveWideningConfig>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            iterations: 10_000,
            exploration_constant: std::f64::consts::SQRT_2,
            max_depth: 50,
            tree_policy: TreePolicy::default(),
            heuristic_weight: 0.35,
            rave: RaveConfig::default(),
            progressive_widening: None,
        }
    }
}

impl SearchConfig {
    /// Creates a builder initialized with the default search configuration.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns a [`SearchConfigBuilder`] seeded with [`SearchConfig::default`].
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::SearchConfig;
    ///
    /// let config = SearchConfig::builder().iterations(256).build();
    /// assert_eq!(config.iterations, 256);
    /// ```
    #[must_use]
    pub fn builder() -> SearchConfigBuilder {
        SearchConfigBuilder(Self::default())
    }

    /// Parses a config from TOML.
    ///
    /// # Parameters
    ///
    /// - `input`: TOML document that matches the [`SearchConfig`] schema.
    ///
    /// # Returns
    ///
    /// Returns the parsed [`SearchConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`toml::de::Error`] when the configuration cannot be parsed.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mctrust::SearchConfig;
    ///
    /// let config = SearchConfig::from_toml_str("iterations = 32").unwrap();
    /// assert_eq!(config.iterations, 32);
    /// ```
    pub fn from_toml_str(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    /// Reads a TOML config file from disk.
    ///
    /// # Parameters
    ///
    /// - `path`: Path to the TOML config file.
    ///
    /// # Returns
    ///
    /// Returns the parsed [`SearchConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchConfigLoadError::Io`] if the file cannot be read,
    /// or [`SearchConfigLoadError::Toml`] if parsing fails.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self, SearchConfigLoadError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(SearchConfigLoadError::Io)?;
        toml::from_str(&contents).map_err(SearchConfigLoadError::Toml)
    }
}

/// Builder for [`SearchConfig`].
pub struct SearchConfigBuilder(SearchConfig);

impl SearchConfigBuilder {
    /// Sets the maximum number of simulations to run.
    ///
    /// # Parameters
    ///
    /// - `iterations`: Simulation budget to store in the builder.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.0.iterations = iterations;
        self
    }

    /// Sets the exploration constant used by the search policy.
    ///
    /// # Parameters
    ///
    /// - `exploration_constant`: Value used to balance exploration against exploitation.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn exploration_constant(mut self, exploration_constant: f64) -> Self {
        self.0.exploration_constant = exploration_constant;
        self
    }

    /// Sets the default rollout depth cap.
    ///
    /// # Parameters
    ///
    /// - `max_depth`: Maximum depth allowed during simulations.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.0.max_depth = max_depth;
        self
    }

    /// Selects the tree policy used during child selection.
    ///
    /// # Parameters
    ///
    /// - `tree_policy`: Policy to store in the builder.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn tree_policy(mut self, tree_policy: TreePolicy) -> Self {
        self.0.tree_policy = tree_policy;
        self
    }

    /// Sets how strongly heuristic estimates influence simulation rewards.
    ///
    /// # Parameters
    ///
    /// - `heuristic_weight`: Blend weight for heuristic scores.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn heuristic_weight(mut self, heuristic_weight: f64) -> Self {
        self.0.heuristic_weight = heuristic_weight;
        self
    }

    /// Sets the RAVE/AMAF configuration.
    ///
    /// # Parameters
    ///
    /// - `rave`: RAVE configuration to store.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn rave(mut self, rave: RaveConfig) -> Self {
        self.0.rave = rave;
        self
    }

    /// Enables progressive widening with the provided settings.
    ///
    /// # Parameters
    ///
    /// - `widening`: Progressive widening configuration to attach.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn progressive_widening(mut self, widening: ProgressiveWideningConfig) -> Self {
        self.0.progressive_widening = Some(widening);
        self
    }

    /// Finalizes the builder and returns the accumulated [`SearchConfig`].
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the built [`SearchConfig`].
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn build(self) -> SearchConfig {
        self.0
    }
}

/// Error returned when loading config from disk.
#[derive(Debug)]
pub enum SearchConfigLoadError {
    /// File read failed.
    Io(std::io::Error),
    /// TOML parsing failed.
    Toml(toml::de::Error),
}

impl std::fmt::Display for SearchConfigLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => {
                write!(
                    f,
                    "failed to read config: {error}. Fix: check file path permissions and ownership, then retry with a readable configuration file."
                )
            }
            Self::Toml(error) => write!(
                f,
                "failed to parse TOML config: {error}. Fix: validate the section layout and ensure key/value types match the expected schema."
            ),
        }
    }
}

impl std::error::Error for SearchConfigLoadError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let c = SearchConfig::default();
        assert_eq!(c.iterations, 10_000);
        assert_eq!(c.tree_policy, TreePolicy::Uct);
        assert!(c.rave.enabled);
    }

    #[test]
    fn builder_overrides() {
        let c = SearchConfig::builder()
            .iterations(500)
            .exploration_constant(3.0)
            .max_depth(10)
            .tree_policy(TreePolicy::Puct { prior_weight: 2.0 })
            .heuristic_weight(0.5)
            .build();

        assert_eq!(c.iterations, 500);
        assert_eq!(c.max_depth, 10);
        assert_eq!(c.tree_policy, TreePolicy::Puct { prior_weight: 2.0 });
    }

    #[test]
    fn parse_from_toml() {
        let config = SearchConfig::from_toml_str(
            r#"
iterations = 64
max_depth = 12

[tree_policy]
kind = "thompson_sampling"
temperature = 0.25
"#,
        )
        .unwrap();

        assert_eq!(config.iterations, 64);
        assert_eq!(
            config.tree_policy,
            TreePolicy::ThompsonSampling { temperature: 0.25 }
        );
    }

    #[test]
    fn progressive_widening_roundtrip() {
        let config = SearchConfig::builder()
            .progressive_widening(ProgressiveWideningConfig {
                minimum_children: 2,
                coefficient: 1.75,
                exponent: 0.4,
            })
            .build();

        let serialized = toml::to_string(&config).unwrap();
        let parsed: SearchConfig = toml::from_str(&serialized).unwrap();

        let widening = parsed.progressive_widening.unwrap();
        assert_eq!(widening.minimum_children, 2);
        assert!((widening.exponent - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_toml_with_all_sections() {
        let config = SearchConfig::from_toml_str(
            r"
iterations = 64
max_depth = 7
heuristic_weight = 0.42

[rave]
enabled = false
bias = 111.0

[progressive_widening]
minimum_children = 2
coefficient = 2.5
exponent = 0.4
",
        )
        .unwrap();

        assert!(!config.rave.enabled);
        assert!((config.rave.bias - 111.0).abs() < f64::EPSILON);
        assert_eq!(
            config
                .progressive_widening
                .as_ref()
                .unwrap()
                .minimum_children,
            2
        );
    }

    #[test]
    fn parse_from_toml_file_error() {
        let err = SearchConfig::from_toml_file("/does/not/exist.toml").unwrap_err();
        assert!(matches!(err, SearchConfigLoadError::Io(_)));
    }

    #[test]
    fn tree_policy_default_is_uct() {
        let policy: TreePolicy = TreePolicy::default();
        assert!(matches!(policy, TreePolicy::Uct));
    }

    #[test]
    fn tree_policy_puct_is_round_trip_toml() {
        let config = SearchConfig::builder()
            .tree_policy(TreePolicy::Puct { prior_weight: 0.8 })
            .build();

        let text = toml::to_string(&config).unwrap();
        let loaded: SearchConfig = toml::from_str(&text).unwrap();
        assert_eq!(loaded.tree_policy, TreePolicy::Puct { prior_weight: 0.8 });
    }

    #[test]
    fn parse_bad_toml_reports_error() {
        let bad = "max_depth = 'oops'";
        assert!(SearchConfig::from_toml_str(bad).is_err());
    }
}
