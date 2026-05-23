use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memex_core::top_k;

fn retrieval_benchmark(c: &mut Criterion) {
    let query = vec![0.25_f32; 384];
    let vectors = vec![vec![0.25_f32; 384]; 10_000];
    c.bench_function("top_k_10k_384d", |b| {
        b.iter(|| top_k(black_box(&query), black_box(&vectors), black_box(10)))
    });
}

criterion_group!(benches, retrieval_benchmark);
criterion_main!(benches);
