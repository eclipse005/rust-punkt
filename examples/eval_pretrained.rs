// Evaluates how well TrainingData::english() pretrained data splits sentences
// against NLTK ground truth.
//
// Usage: cargo run --release --example eval_pretrained
//
// Metric: set-based precision / recall / F1 (not position-aligned).
//
// This exercises the pretrained path -- which is what voxtrans actually runs.
// Compare to sentence_tokenizer_compare_nltk_train_on_document in the unit
// tests, which uses the train-on-document path.

use std::collections::HashSet;

use punkt::params::Standard;
use punkt::{SentenceTokenizer, TrainingData};

struct Case {
  name: &'static str,
  raw: &'static str,
  expected: &'static str,
}

const CASES: &[Case] = &[
  Case {
    name: "npr-article-01",
    raw: include_str!("../test/raw/npr-article-01.txt"),
    expected: include_str!("../test/sentence/npr-article-01.txt"),
  },
  Case {
    name: "ny-times-article-01",
    raw: include_str!("../test/raw/ny-times-article-01.txt"),
    expected: include_str!("../test/sentence/ny-times-article-01.txt"),
  },
  Case {
    name: "pride-and-prejudice",
    raw: include_str!("../test/raw/pride-and-prejudice.txt"),
    expected: include_str!("../test/sentence/pride-and-prejudice.txt"),
  },
  Case {
    name: "history-of-china",
    raw: include_str!("../test/raw/history-of-china.txt"),
    expected: include_str!("../test/sentence/history-of-china.txt"),
  },
  Case {
    name: "the-sayings-of-confucius",
    raw: include_str!("../test/raw/the-sayings-of-confucius.txt"),
    expected: include_str!("../test/sentence/the-sayings-of-confucius.txt"),
  },
];

// NLTK ground truth wraps each sentence in [...].
fn normalize_expected(line: &str) -> String {
  line.trim().to_string()
}

// rust-punkt's SentenceTokenizer yields &str; wrap to the same [...] form
// for set-based comparison. Mirrors the normalize rules in tokenizer.rs tests.
fn normalize_detected(s: &str) -> String {
  format!("[{}]", s)
    .replace('"', "\\\"")
    .replace('\n', "\\n")
    .replace('\r', "")
}

fn main() {
  let data = TrainingData::english();

  println!(
    "{:<26} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7} {:>7}",
    "document", "|D|", "|E|", "TP", "FN", "P%", "R%", "F1%"
  );
  println!("{}", "-".repeat(80));

  let mut total_d = 0usize;
  let mut total_e = 0usize;
  let mut total_tp = 0usize;
  let mut total_fn = 0usize;

  for case in CASES {
    let sentences: Vec<&str> = SentenceTokenizer::<Standard>::new(case.raw, &data).collect();
    let expected: Vec<String> = case
      .expected
      .lines()
      .filter(|s| !s.trim().is_empty())
      .map(normalize_expected)
      .collect();

    let detected: HashSet<String> = sentences.iter().map(|s| normalize_detected(s)).collect();
    let expected_set: HashSet<String> = expected.iter().cloned().collect();

    // Set-theoretic metrics
    let tp = detected.intersection(&expected_set).count();
    let fn_ = expected_set.difference(&detected).count();

    let d = detected.len();
    let e = expected_set.len();
    let p = if d > 0 {
      100.0 * tp as f64 / d as f64
    } else {
      0.0
    };
    let r = if e > 0 {
      100.0 * tp as f64 / e as f64
    } else {
      0.0
    };
    let f1 = if p + r > 0.0 {
      2.0 * p * r / (p + r)
    } else {
      0.0
    };

    println!(
      "{:<26} {:>6} {:>6} {:>6} {:>6} {:>6.2}% {:>6.2}% {:>6.2}%",
      case.name, d, e, tp, fn_, p, r, f1
    );

    total_d += d;
    total_e += e;
    total_tp += tp;
    total_fn += fn_;
  }

  println!("{}", "-".repeat(80));
  let p = if total_d > 0 {
    100.0 * total_tp as f64 / total_d as f64
  } else {
    0.0
  };
  let r = if total_e > 0 {
    100.0 * total_tp as f64 / total_e as f64
  } else {
    0.0
  };
  let f1 = if p + r > 0.0 {
    2.0 * p * r / (p + r)
  } else {
    0.0
  };
  println!(
    "{:<26} {:>6} {:>6} {:>6} {:>6} {:>6.2}% {:>6.2}% {:>6.2}%",
    "TOTAL (micro-avg)", total_d, total_e, total_tp, total_fn, p, r, f1
  );
}
