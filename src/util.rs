// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::prelude::DefinesSentenceEndings;
use crate::token::Token;
use crate::trainer::TrainingData;

/// Peforms a first pass annotation on a Token.
pub fn annotate_first_pass<P: DefinesSentenceEndings>(tok: &Token, data: &TrainingData) {
  let is_split_abbrev = tok
    .tok_without_period()
    .split('-')
    .next_back()
    .map(|s| data.contains_abbrev(s))
    .unwrap_or(false);

  if tok.tok().len() == 1 && P::is_sentence_ending(&tok.tok().chars().next().unwrap()) {
    tok.set_is_sentence_break(true);
  } else if tok.has_final_period() && !tok.is_ellipsis() {
    if is_split_abbrev || data.contains_abbrev(tok.tok_without_period()) {
      tok.set_is_abbrev(true);
    } else {
      tok.set_is_sentence_break(true);
    }
  }
}

pub fn dunning_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
  // Clamp probabilities away from 0 and 1 to avoid log(0) = -inf producing NaN
  // when multiplied by a zero count. Mirrors the epsilon guard NLTK added in
  // https://github.com/nltk/nltk/pull/3244 (2023-10). For normal inputs (where
  // the probability already lies strictly inside (0, 1)) the clamp is a no-op.
  const EPSILON: f64 = 1e-10;
  let p1 = (count_b / n).clamp(EPSILON, 1.0 - EPSILON);
  let p2: f64 = 0.99;
  let nullh = count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln();
  let alth = count_ab * p2.ln() + (count_a - count_ab) * (1.0 - p2).ln();

  -2.0 * (nullh - alth)
}

pub fn col_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
  const EPSILON: f64 = 1e-10;
  let p = (count_b / n).clamp(EPSILON, 1.0 - EPSILON);
  let p1 = (count_ab / count_a).clamp(EPSILON, 1.0 - EPSILON);
  let p2 = ((count_b - count_ab) / (n - count_a)).clamp(EPSILON, 1.0 - EPSILON);

  let s1 = count_ab * p.ln() + (count_a - count_ab) * (1.0 - p).ln();
  let s2 = (count_b - count_ab) * p.ln() + (n - count_a - count_b + count_ab) * (1.0 - p).ln();
  let s3 = if count_a == count_ab {
    0f64
  } else {
    count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln()
  };
  let s4 = if count_b == count_ab {
    0f64
  } else {
    (count_b - count_ab) * p2.ln() + (n - count_a - count_b + count_ab) * (1.0 - p2).ln()
  };

  -2.0 * (s1 + s2 - s3 - s4)
}
