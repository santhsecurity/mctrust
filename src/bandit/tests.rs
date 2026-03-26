use super::*;

#[test]
fn bandit_basic_flow() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 42);

    for i in 0..20u64 {
        search.add_arm(i, (i / 5) as u32);
    }

    let mut pulled = Vec::new();
    for _ in 0..20 {
        if let Some(arm) = search.next_arm() {
            pulled.push(arm);
            let reward = if arm < 5 { 1.0 } else { 0.0 };
            search.observe(arm, reward);
        } else {
            break;
        }
    }

    assert_eq!(pulled.len(), 20);
}

#[test]
fn bandit_respects_budget() {
    let config = BanditConfig::builder().max_pulls(5).build();
    let mut search = BanditSearch::new_seeded(config, 42);

    for i in 0..50u64 {
        search.add_arm(i, 0);
    }

    let mut count = 0;
    while let Some(arm) = search.next_arm() {
        count += 1;
        search.observe(arm, 1.0);
        if count > 100 {
            panic!("budget not respected");
        }
    }

    assert_eq!(count, 5);
}

#[test]
fn bandit_empty_returns_none() {
    let mut search = BanditSearch::new(BanditConfig::default());
    assert!(search.next_arm().is_none());
}

#[test]
fn bandit_rave_propagates() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 99);

    // Two groups, 5 arms each.
    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }

    // Pull two arms with positive reward so RAVE definitely fires.
    let arm1 = search.next_arm().unwrap();
    search.observe(arm1, 1.0);
    let arm2 = search.next_arm().unwrap();
    search.observe(arm2, 1.0);

    // At least one group's node should have RAVE visits from the other.
    let stats = search.group_stats();
    let total_rave: u32 = stats.iter().map(|s| s.rave_visits).sum();
    assert_eq!(total_rave, 2);
}

#[test]
fn bandit_group_bias() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 7);

    for i in 0..10u64 {
        search.add_arm(i, (i / 5) as u32);
    }

    // Bias group 1 heavily.
    search.set_group_bias(1, 100.0);

    // First pull should come from group 1 due to the massive bias.
    let arm = search.next_arm().unwrap();
    assert!(arm >= 5, "biased group should be selected first");
}

#[test]
fn bandit_total_pulls() {
    let mut search = BanditSearch::new(BanditConfig::default());
    for i in 0..5u64 {
        search.add_arm(i, 0);
    }

    for _ in 0..3 {
        if let Some(arm) = search.next_arm() {
            search.observe(arm, 0.5);
        }
    }

    assert_eq!(search.total_pulls(), 3);
}

#[test]
fn bandit_group_stats_correct() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 1);

    for i in 0..6u64 {
        search.add_arm(i, (i / 3) as u32);
    }

    // Pull 3 arms, all from whatever UCT selects.
    for _ in 0..3 {
        if let Some(arm) = search.next_arm() {
            search.observe(arm, 0.5);
        }
    }

    let stats = search.group_stats();
    assert_eq!(stats.len(), 2);

    let total_visits: u32 = stats.iter().map(|s| s.visits).sum();
    assert_eq!(total_visits, 3);
}

#[test]
fn bandit_seeded_determinism() {
    let config = BanditConfig::default();

    let mut s1 = BanditSearch::new_seeded(config.clone(), 42);
    let mut s2 = BanditSearch::new_seeded(config, 42);

    for i in 0..10u64 {
        s1.add_arm(i, (i / 5) as u32);
        s2.add_arm(i, (i / 5) as u32);
    }

    for _ in 0..10 {
        let a1 = s1.next_arm();
        let a2 = s2.next_arm();
        assert_eq!(a1, a2, "same seed should produce same sequence");
        if let (Some(arm1), Some(arm2)) = (a1, a2) {
            s1.observe(arm1, 0.5);
            s2.observe(arm2, 0.5);
        }
    }
}

#[test]
fn bandit_many_groups() {
    let mut search = BanditSearch::new(BanditConfig::default());

    // 100 groups, 10 arms each.
    for group in 0..100u32 {
        for arm in 0..10u32 {
            search.add_arm(u64::from(group) * 10 + u64::from(arm), group);
        }
    }

    // Should be able to pull at least 100 arms.
    let mut count = 0;
    for _ in 0..200 {
        if let Some(arm) = search.next_arm() {
            search.observe(arm, 0.1);
            count += 1;
        } else {
            break;
        }
    }

    assert!(count >= 100);
}

#[test]
fn bandit_checkpoint_roundtrip() {
    let mut search = BanditSearch::new_seeded(BanditConfig::default(), 11);
    for i in 0..7u64 {
        search.add_arm(i, 0);
    }
    for _ in 0..3 {
        if let Some(arm) = search.next_arm() {
            search.observe(arm, 0.75);
        }
    }

    let checkpoint = search.checkpoint();
    let restored = BanditSearch::restore(checkpoint);
    assert_eq!(restored.total_pulls(), 3);
}

#[test]
fn bandit_prevents_duplicate_arms() {
    let mut search = BanditSearch::new(BanditConfig::default());
    search.add_arm(1, 0);
    search.add_arm(1, 0);
    let stats = search.group_stats();

    assert_eq!(stats[0].total_arms, 1);
}

#[test]
fn bandit_zero_max_pulls_disables_limit() {
    let config = BanditConfig::builder().max_pulls(0).build();
    let mut search = BanditSearch::new_seeded(config, 7);

    for i in 0..15u64 {
        search.add_arm(i, 0);
    }

    for _ in 0..15 {
        if search.next_arm().is_none() {
            break;
        }
    }

    assert_eq!(search.total_pulls(), 15);
}
