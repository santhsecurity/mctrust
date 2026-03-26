//! Adversarial edge case tests for mctrust
//!
//! These tests are designed to break the crate by testing:
//! - Zero arms
//! - Single arm
//! - Explore constant of 0
//! - Explore constant of infinity
//! - NaN rewards
//! - Negative rewards
//! - Concurrent observe calls
//! - Max u32 visits

use crate::*;
use std::sync::Arc;
use std::thread;

// =============================================================================
// Zero Arms Tests
// =============================================================================

#[test]
fn bandit_with_zero_arms() {
    let mut search = BanditSearch::new(BanditConfig::default());
    
    // No arms added
    let arm = search.next_arm();
    
    assert!(arm.is_none(), "Bandit with no arms should return None");
}

#[test]
fn bandit_group_stats_with_zero_arms() {
    let search = BanditSearch::new(BanditConfig::default());
    
    let stats = search.group_stats();
    
    assert!(stats.is_empty(), "Stats should be empty with no arms");
}

#[test]
fn bandit_total_pulls_with_zero_arms() {
    let search = BanditSearch::new(BanditConfig::default());
    
    assert_eq!(search.total_pulls(), 0, "Total pulls should be 0");
}

// =============================================================================
// Single Arm Tests
// =============================================================================

#[test]
fn bandit_with_single_arm() {
    let mut search = BanditSearch::new(BanditConfig::default());
    search.add_arm(0, 0);
    
    let arm1 = search.next_arm();
    assert_eq!(arm1, Some(0), "Should return the single arm");
    
    // After pulling the only arm, should return None
    let arm2 = search.next_arm();
    assert!(arm2.is_none(), "Should return None after single arm is exhausted");
}

#[test]
fn bandit_single_arm_with_observe() {
    let mut search = BanditSearch::new(BanditConfig::default());
    search.add_arm(42, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, 1.0);
    
    let stats = search.group_stats();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].visits, 1);
    assert_eq!(stats[0].average_reward, 1.0);
}

// =============================================================================
// Explore Constant Tests
// =============================================================================

#[test]
fn bandit_exploration_constant_zero() {
    let config = BanditConfig::builder()
        .exploration_constant(0.0)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 3) as u32);
    }
    
    // With exploration constant 0, it should still work (pure exploitation after first visit)
    let arm = search.next_arm();
    assert!(arm.is_some());
    
    search.observe(arm.unwrap(), 0.5);
}

#[test]
fn bandit_exploration_constant_infinity() {
    let config = BanditConfig::builder()
        .exploration_constant(f64::INFINITY)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 3) as u32);
    }
    
    // With infinite exploration constant, behavior depends on implementation
    // The UCT formula should handle this (exploration term will dominate)
    let arm = search.next_arm();
    assert!(arm.is_some());
}

#[test]
fn bandit_exploration_constant_nan() {
    let config = BanditConfig::builder()
        .exploration_constant(f64::NAN)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 3) as u32);
    }
    
    // With NaN exploration constant, behavior is undefined
    // This may panic or return unexpected results
    let _arm = search.next_arm();
    // The result depends on how NaN propagates through the UCT formula
}

#[test]
fn bandit_exploration_constant_negative() {
    let config = BanditConfig::builder()
        .exploration_constant(-1.0)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 3) as u32);
    }
    
    // Negative exploration constant should still work (exploration becomes negative)
    let arm = search.next_arm();
    assert!(arm.is_some());
}

// =============================================================================
// RAVE Bias Tests
// =============================================================================

#[test]
fn bandit_rave_bias_zero() {
    let config = BanditConfig::builder()
        .rave_bias(0.0)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }
    
    let arm1 = search.next_arm().unwrap();
    search.observe(arm1, 1.0);
    
    let arm2 = search.next_arm().unwrap();
    search.observe(arm2, 1.0);
    
    // Fixed: RAVE updates are now skipped when rave_bias=0
    let stats = search.group_stats();
    let total_rave: u32 = stats.iter().map(|s| s.rave_visits).sum();
    assert_eq!(total_rave, 0, "RAVE visits should not increment when rave_bias=0");
}

#[test]
fn bandit_rave_bias_infinity() {
    let config = BanditConfig::builder()
        .rave_bias(f64::INFINITY)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }
    
    let arm = search.next_arm();
    assert!(arm.is_some());
}

// =============================================================================
// NaN and Infinity Reward Tests
// =============================================================================

#[test]
fn bandit_observe_nan_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, f64::NAN);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1);
    // Average reward with NaN should be NaN
    assert!(stats[0].average_reward.is_nan(), "Average should be NaN when reward is NaN");
}

#[test]
fn bandit_observe_infinity_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, f64::INFINITY);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1);
    assert!(stats[0].average_reward.is_infinite(), "Average should be infinite");
}

#[test]
fn bandit_observe_negative_infinity_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, f64::NEG_INFINITY);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1);
    assert!(stats[0].average_reward.is_infinite() && stats[0].average_reward < 0.0,
            "Average should be negative infinite");
}

#[test]
fn bandit_observe_negative_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, -5.0);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1);
    assert_eq!(stats[0].average_reward, -5.0, "Negative reward should be accepted");
}

#[test]
fn bandit_mixed_rewards_positive_and_negative() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    for i in 0..5u64 {
        search.add_arm(i, 0);
    }
    
    // Pull each arm with different rewards
    for i in 0..5u64 {
        let arm = search.next_arm().unwrap();
        let reward = if i % 2 == 0 { 1.0 } else { -1.0 };
        search.observe(arm, reward);
    }
    
    let stats = search.group_stats();
    assert_eq!(stats[0].total_arms, 5);
}

#[test]
fn bandit_observe_very_large_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, 1e308); // Very large but not quite infinity
    
    let stats = search.group_stats();
    assert!(stats[0].average_reward > 1e307, "Very large reward should be preserved");
}

#[test]
fn bandit_observe_very_small_reward() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    let arm = search.next_arm().unwrap();
    search.observe(arm, 1e-308); // Very small positive
    
    let stats = search.group_stats();
    assert!(stats[0].average_reward > 0.0 && stats[0].average_reward < 1e-307,
            "Very small reward should be preserved");
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[test]
fn bandit_concurrent_observe_calls() {
    let search = Arc::new(std::sync::Mutex::new(BanditSearch::new_seeded(BanditConfig::default(), 42)));
    
    // Add 200 arms so all threads can pull
    {
        let mut s = search.lock().unwrap();
        for i in 0..200u64 {
            s.add_arm(i, (i / 50) as u32);
        }
    }
    
    let mut handles = vec![];
    
    for thread_id in 0..10 {
        let search = Arc::clone(&search);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                let mut s = search.lock().unwrap();
                if let Some(arm) = s.next_arm() {
                    s.observe(arm, thread_id as f64 * 0.1);
                }
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let s = search.lock().unwrap();
    assert_eq!(s.total_pulls(), 100, "Should have 100 total pulls");
}

// =============================================================================
// Max Visits Tests
// =============================================================================

#[test]
fn bandit_many_visits_u32_boundary() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    // Simulate many observations (not actually u32::MAX to keep test fast)
    for _ in 0..1000 {
        // We need to pull the arm first, then observe
        // But after first pull, next_arm returns None for exhausted arms
        // So we observe directly on arm 0
        search.observe(0, 0.5);
    }
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1000);
}

// =============================================================================
// Group Bias Tests
// =============================================================================

#[test]
fn bandit_group_bias_infinity() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }
    
    search.set_group_bias(1, f64::INFINITY);
    
    // First pull should come from group 1 due to infinite bias
    let arm = search.next_arm().unwrap();
    assert!(arm >= 5, "Infinite bias should force selection from group 1");
}

#[test]
fn bandit_group_bias_nan() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }
    
    search.set_group_bias(0, f64::NAN);
    
    // Behavior with NaN bias depends on implementation
    let arm = search.next_arm();
    assert!(arm.is_some());
}

#[test]
fn bandit_set_bias_for_nonexistent_group() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    // Setting bias for a group that doesn't exist should not panic
    search.set_group_bias(999, 100.0);
    
    let arm = search.next_arm();
    assert!(arm.is_some());
}

// =============================================================================
// Max Pulls Tests
// =============================================================================

#[test]
fn bandit_max_pulls_of_one() {
    let config = BanditConfig::builder()
        .max_pulls(1)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    for i in 0..10u64 {
        search.add_arm(i, 0);
    }
    
    let arm1 = search.next_arm();
    assert!(arm1.is_some());
    
    let arm2 = search.next_arm();
    assert!(arm2.is_none(), "Should return None after max_pulls reached");
}

#[test]
fn bandit_max_pulls_u64_max() {
    let config = BanditConfig::builder()
        .max_pulls(u64::MAX)
        .build();
    
    let mut search = BanditSearch::new_seeded(config, 42);
    
    search.add_arm(0, 0);
    
    // With max_pulls = u64::MAX, we should be able to pull many times
    for _ in 0..100 {
        let arm = search.next_arm();
        if arm.is_none() {
            break; // Arm exhausted
        }
    }
    
    // Total pulls should not exceed the number of unique arms
    assert!(search.total_pulls() <= 1);
}

// =============================================================================
// Checkpoint Tests
// =============================================================================

#[test]
fn bandit_checkpoint_empty() {
    let search = BanditSearch::new(BanditConfig::default());
    
    let checkpoint = search.checkpoint();
    let mut restored = BanditSearch::restore(checkpoint);
    
    assert_eq!(restored.total_pulls(), 0);
    assert!(restored.next_arm().is_none());
}

#[test]
fn bandit_checkpoint_with_observations() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    for i in 0..10u64 {
        search.add_arm(i, 0);
    }
    
    // Pull some arms
    for _ in 0..5 {
        let arm = search.next_arm().unwrap();
        search.observe(arm, 1.0);
    }
    
    let checkpoint = search.checkpoint();
    let restored = BanditSearch::restore(checkpoint);
    
    assert_eq!(restored.total_pulls(), 5);
}

// =============================================================================
// Add Arm Edge Cases
// =============================================================================

#[test]
fn bandit_add_duplicate_arm() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    search.add_arm(0, 0); // Duplicate
    
    let stats = search.group_stats();
    assert_eq!(stats[0].total_arms, 1, "Duplicate arm should not be added");
}

#[test]
fn bandit_add_arm_with_u32_max_group() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, u32::MAX);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].group_id, u32::MAX);
}

#[test]
fn bandit_add_arm_with_u64_max_arm_id() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(u64::MAX, 0);
    
    let arm = search.next_arm();
    assert_eq!(arm, Some(u64::MAX));
}

#[test]
fn bandit_add_many_groups() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    // Add 1000 groups with 1 arm each
    for i in 0..1000u64 {
        search.add_arm(i, i as u32);
    }
    
    let stats = search.group_stats();
    assert_eq!(stats.len(), 1000);
}

#[test]
fn bandit_add_many_arms_single_group() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    // Add 1000 arms to a single group
    for i in 0..1000u64 {
        search.add_arm(i, 0);
    }
    
    let stats = search.group_stats();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].total_arms, 1000);
}

// =============================================================================
// Observe Nonexistent Arm Tests
// =============================================================================

#[test]
fn bandit_observe_nonexistent_arm() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    // Observing an arm that was never pulled (and doesn't exist in mapping)
    // This should not panic
    search.observe(999, 1.0);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 0, "Nonexistent arm observation should not affect stats");
}

#[test]
fn bandit_observe_before_next_arm() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);
    
    search.add_arm(0, 0);
    
    // Observing before pulling - arm exists in mapping
    search.observe(0, 1.0);
    
    let stats = search.group_stats();
    assert_eq!(stats[0].visits, 1);
}
