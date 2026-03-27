use serde::{Deserialize, Serialize};

/// Configuration for [`crate::BanditSearch`].
///
/// # Examples
///
/// ```
/// use mctrust::BanditConfig;
///
/// let config = BanditConfig::builder()
///     .exploration_constant(2.0)
///     .rave_bias(300.0)
///     .max_pulls(10_000)
///     .build();
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BanditConfig {
    /// UCT exploration constant. Default: `sqrt(2)`.
    pub exploration_constant: f64,

    /// RAVE bias weight. Controls cross-arm reward propagation.
    ///
    /// When an arm in group A receives a high reward, all unvisited arms
    /// in other groups get a proportional RAVE boost. The bias decays
    /// as visit counts increase.
    ///
    /// - `0.0` = RAVE disabled (pure UCT).
    /// - `500.0` = strong RAVE influence early, decays with visits.
    ///
    /// Default: `500.0`.
    pub rave_bias: f64,

    /// Maximum number of arms to pull before stopping. `0` = unlimited.
    pub max_pulls: u64,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
            rave_bias: 500.0,
            max_pulls: 0,
        }
    }
}

impl BanditConfig {
    /// Creates a builder initialized with default bandit-search settings.
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns a [`BanditConfigBuilder`] seeded from [`BanditConfig::default`].
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn builder() -> BanditConfigBuilder {
        BanditConfigBuilder(Self::default())
    }
}

/// Fluent builder for [`BanditConfig`].
pub struct BanditConfigBuilder(BanditConfig);

impl BanditConfigBuilder {
    /// Sets the UCT exploration constant.
    ///
    /// # Parameters
    ///
    /// - `c`: Exploration multiplier to store in the builder.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn exploration_constant(mut self, c: f64) -> Self {
        self.0.exploration_constant = c;
        self
    }

    /// Sets the RAVE bias weight.
    ///
    /// # Parameters
    ///
    /// - `bias`: RAVE bias value to store in the builder.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn rave_bias(mut self, bias: f64) -> Self {
        self.0.rave_bias = bias;
        self
    }

    /// Sets the maximum number of arm pulls.
    ///
    /// # Parameters
    ///
    /// - `n`: Pull budget to enforce, or `0` for unlimited.
    ///
    /// # Returns
    ///
    /// Returns the updated builder.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn max_pulls(mut self, n: u64) -> Self {
        self.0.max_pulls = n;
        self
    }

    /// Finalizes the builder and returns the accumulated [`BanditConfig`].
    ///
    /// # Parameters
    ///
    /// This function takes no additional parameters.
    ///
    /// # Returns
    ///
    /// Returns the built [`BanditConfig`].
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn build(self) -> BanditConfig {
        self.0
    }
}
