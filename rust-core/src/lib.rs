use std::collections::{HashMap, HashSet};

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

pub fn top_k(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| (index, cosine_similarity(query, vector)))
        .collect();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    scored.truncate(k);
    scored
}

pub fn summarize_sentences(texts: &[String], max_sentences: usize) -> String {
    let mut seen = HashSet::new();
    let mut sentences = Vec::new();
    for text in texts {
        for chunk in text.replace('\n', " ").split('.') {
            let sentence = chunk.trim();
            if sentence.is_empty() {
                continue;
            }
            let key = sentence.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
            if seen.insert(key) {
                sentences.push(format!("{}.", sentence.trim_end_matches('.')));
            }
            if sentences.len() >= max_sentences {
                return sentences.join(" ");
            }
        }
    }
    sentences.join(" ")
}

pub fn diff_ids(local: &[String], remote: &[String]) -> Vec<String> {
    let local_set: HashSet<&String> = local.iter().collect();
    remote
        .iter()
        .filter(|id| !local_set.contains(id))
        .cloned()
        .collect()
}

pub fn merge_lww(
    left: &HashMap<String, i64>,
    right: &HashMap<String, i64>,
) -> HashMap<String, i64> {
    let mut merged = left.clone();
    for (key, value) in right {
        let current = merged.get(key).copied().unwrap_or(i64::MIN);
        if *value >= current {
            merged.insert(key.clone(), *value);
        }
    }
    merged
}

#[cfg(feature = "python")]
mod python {
    use super::{cosine_similarity, diff_ids, summarize_sentences};
    use pyo3::prelude::*;

    #[pyfunction]
    fn cosine(left: Vec<f32>, right: Vec<f32>) -> f32 {
        cosine_similarity(&left, &right)
    }

    #[pyfunction]
    fn summarize(texts: Vec<String>, max_sentences: usize) -> String {
        summarize_sentences(&texts, max_sentences)
    }

    #[pyfunction]
    fn sync_diff(local: Vec<String>, remote: Vec<String>) -> Vec<String> {
        diff_ids(&local, &remote)
    }

    #[pymodule]
    fn memex_core(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(cosine, module)?)?;
        module.add_function(wrap_pyfunction!(summarize, module)?)?;
        module.add_function(wrap_pyfunction!(sync_diff, module)?)?;
        Ok(())
    }
}

#[cfg(feature = "node")]
mod node {
    use super::{cosine_similarity, diff_ids, summarize_sentences};
    use napi_derive::napi;

    #[napi]
    pub fn cosine(left: Vec<f32>, right: Vec<f32>) -> f32 {
        cosine_similarity(&left, &right)
    }

    #[napi]
    pub fn summarize(texts: Vec<String>, max_sentences: u32) -> String {
        summarize_sentences(&texts, max_sentences as usize)
    }

    #[napi]
    pub fn sync_diff(local: Vec<String>, remote: Vec<String>) -> Vec<String> {
        diff_ids(&local, &remote)
    }
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, diff_ids, summarize_sentences};

    #[test]
    fn cosine_scores_identical_vectors() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    }

    #[test]
    fn summarize_deduplicates_sentences() {
        let text = vec![
            "User likes local-first apps. User likes local-first apps.".to_string(),
            "User prefers concise answers.".to_string(),
        ];
        assert_eq!(
            summarize_sentences(&text, 4),
            "User likes local-first apps. User prefers concise answers."
        );
    }

    #[test]
    fn diff_finds_remote_only_ids() {
        assert_eq!(
            diff_ids(&["a".to_string()], &["a".to_string(), "b".to_string()]),
            vec!["b".to_string()]
        );
    }
}
