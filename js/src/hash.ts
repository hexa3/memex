function tokenize(text: string): string[] {
  return Array.from(text.toLowerCase().matchAll(/[a-z0-9][a-z0-9_'-]*/g), (match) => match[0]);
}

function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) return vector;
  return vector.map((value) => value / norm);
}

export function embedHash(text: string, dimension = 384): number[] {
  if (dimension < 16) throw new Error("dimension must be at least 16");
  const vector = Array.from({ length: dimension }, () => 0);
  const tokens = tokenize(text);
  const features = [...tokens];
  for (let index = 0; index < tokens.length - 1; index += 1) {
    features.push(`${tokens[index]} ${tokens[index + 1]}`);
  }
  if (features.length === 0) features.push(text.toLowerCase());
  for (const feature of features) {
    const primary = hashFeature(feature, 0x811c9dc5);
    const secondary = hashFeature(feature, 0x9e3779b9);
    const bucket = primary % dimension;
    const sign = secondary & 1 ? 1 : -1;
    const weight = 1 + Math.min(feature.length, 24) / 48;
    vector[bucket] = (vector[bucket] ?? 0) + sign * weight;
  }
  return normalize(vector);
}

function hashFeature(value: string, seed: number): number {
  let hash = seed >>> 0;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length !== right.length) throw new Error("embedding dimension mismatch");
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (leftNorm === 0 || rightNorm === 0) return 0;
  return dot / Math.sqrt(leftNorm * rightNorm);
}
