// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Minimal frequency distribution type used by the trainer.
//!
//! This replaces the deprecated `rust-freqdist` crate. The API mirrors the
//! subset of `freqdist::FrequencyDistribution` that the trainer relies on.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;

/// A frequency distribution tracks how many times each unique key has been
/// observed, along with the total number of observations.
pub struct FrequencyDistribution<K> {
  counts: HashMap<K, usize>,
  total: usize,
}

impl<K: Eq + Hash> FrequencyDistribution<K> {
  /// Creates a new, empty frequency distribution.
  #[inline(always)]
  pub fn new() -> FrequencyDistribution<K> {
    FrequencyDistribution {
      counts: HashMap::new(),
      total: 0,
    }
  }

  /// Records another observation of `key`.
  #[inline]
  pub fn insert(&mut self, key: K) {
    *self.counts.entry(key).or_insert(0) += 1;
    self.total += 1;
  }

  /// Returns the number of times `key` has been observed. Returns `0` for
  /// keys that have never been inserted.
  #[inline(always)]
  pub fn get<Q>(&self, key: &Q) -> usize
  where
    K: Borrow<Q>,
    Q: ?Sized + Hash + Eq,
  {
    *self.counts.get(key).unwrap_or(&0)
  }

  /// Iterates over the keys that have been observed at least once.
  #[inline(always)]
  pub fn keys(&self) -> impl Iterator<Item = &K> {
    self.counts.keys()
  }

  /// Returns the total number of observations (the sum of all counts).
  #[inline(always)]
  pub fn sum_counts(&self) -> usize {
    self.total
  }
}

impl<K: Eq + Hash> Default for FrequencyDistribution<K> {
  fn default() -> Self {
    Self::new()
  }
}

impl<K: Eq + Hash> Index<K> for FrequencyDistribution<K> {
  type Output = usize;

  #[inline(always)]
  fn index(&self, index: K) -> &usize {
    static ZERO: usize = 0;
    self.counts.get(&index).unwrap_or(&ZERO)
  }
}
