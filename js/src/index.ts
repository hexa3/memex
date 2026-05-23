import { cosineSimilarity, embedHash } from "./hash.js";
import { HttpClient } from "./http.js";
import { LocalStore } from "./store.js";
import type { ExportPayload, MemoryOptions, MemoryRecord, Metadata, SaveOptions, SearchOptions } from "./types.js";
import { metadataMatches, normalizeNamespace, normalizeText, validateMetadata } from "./validation.js";

function now(): number {
  return Math.floor(Date.now() / 1000);
}

function randomId(): string {
  const cryptoLike = globalThis.crypto as Crypto | undefined;
  if (cryptoLike?.randomUUID) return cryptoLike.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const value = Math.floor(Math.random() * 16);
    const nibble = char === "x" ? value : (value & 0x3) | 0x8;
    return nibble.toString(16);
  });
}

function effectiveImportance(record: MemoryRecord): number {
  const ageDays = Math.max(0, (now() - record.createdAt) / 86_400);
  const recencyFactor = Math.exp(-0.01 * ageDays);
  const accessFactor = 1 + 0.1 * Math.log1p(record.accessCount);
  return Math.max(0, record.importance * recencyFactor * accessFactor);
}

function isExpired(record: MemoryRecord): boolean {
  return record.ttlDays !== undefined && record.createdAt + record.ttlDays * 86_400 <= now();
}

function extractFacts(userMsg: string): string[] {
  const patterns: Array<[RegExp, (value: string) => string]> = [
    [/\bmy name is ([^.!?\n]{1,120})/gi, (value) => `User's name is ${value}.`],
    [/\bi live in ([^.!?\n]{1,120})/gi, (value) => `User lives in ${value}.`],
    [/\bi prefer ([^.!?\n]{1,120})/gi, (value) => `User prefers ${value}.`],
    [/\bi like ([^.!?\n]{1,120})/gi, (value) => `User likes ${value}.`],
    [/\bremember that ([^.!?\n]{1,160})/gi, (value) => `${value[0]?.toUpperCase() ?? ""}${value.slice(1)}.`],
    [/\bmy goal is ([^.!?\n]{1,160})/gi, (value) => `User's goal is ${value}.`],
  ];
  const facts: string[] = [];
  for (const [pattern, format] of patterns) {
    for (const match of userMsg.matchAll(pattern)) {
      const value = (match[1] ?? "").trim().replace(/[.!?]+$/, "");
      if (!value) continue;
      const fact = format(value);
      if (!facts.includes(fact)) facts.push(fact);
      if (facts.length >= 3) return facts;
    }
  }
  return facts;
}

export class Memory {
  private readonly namespace: string;
  private readonly dimension: number;
  private readonly autoDedupe: boolean;
  private readonly dedupeThreshold: number;
  private readonly maxMemories: number;
  private readonly store: LocalStore;
  private readonly http: HttpClient | undefined;

  constructor(options: MemoryOptions = {}) {
    this.namespace = normalizeNamespace(options.namespace);
    this.dimension = options.dimension ?? 384;
    this.autoDedupe = options.autoDedupe ?? true;
    this.dedupeThreshold = options.dedupeThreshold ?? 0.92;
    this.maxMemories = options.maxMemories ?? 10_000;
    this.store = new LocalStore(options.storagePath);
    this.http = options.endpoint ? new HttpClient(options.endpoint) : undefined;
  }

  async save(text: string, options: SaveOptions = {}): Promise<string> {
    const value = normalizeText(text);
    const namespace = normalizeNamespace(options.namespace ?? this.namespace);
    const metadata = validateMetadata(options.metadata);
    if (this.http) return this.http.save(value, { ...options, metadata, namespace });
    const records = await this.store.load();
    const record: MemoryRecord = {
      id: randomId(),
      namespace,
      text: value,
      embedding: embedHash(value.slice(0, 20_000), this.dimension),
      metadata,
      memoryType: options.memoryType ?? "long_term",
      sourceIds: options.sourceIds ?? [],
      importance: options.importance ?? 1,
      createdAt: now(),
      accessedAt: now(),
      accessCount: 0,
      ttlDays: options.ttlDays,
    };
    records.push(record);
    if (this.autoDedupe) this.dedupe(records, record);
    this.prune(records, namespace);
    await this.store.save(records);
    return record.id;
  }

  async recall(query: string): Promise<string | null> {
    if (this.http) return this.http.recall(query);
    const results = await this.search(query, { k: 1 });
    return results[0]?.text ?? null;
  }

  async search(query: string, kOrOptions: number | SearchOptions = 5): Promise<MemoryRecord[]> {
    const options: SearchOptions = typeof kOrOptions === "number" ? { k: kOrOptions } : kOrOptions;
    const value = normalizeText(query, 20_000);
    if (this.http) return this.http.search(value, options);
    const namespace = normalizeNamespace(options.namespace ?? this.namespace);
    const embedding = embedHash(value.slice(0, 20_000), this.dimension);
    const filters = validateMetadata(options.filters);
    const threshold = options.threshold ?? 0;
    const records = await this.store.load();
    const results = records
      .filter((record) => record.namespace === namespace)
      .filter((record) => !isExpired(record))
      .filter((record) => !options.memoryTypes || options.memoryTypes.includes(record.memoryType ?? "long_term"))
      .filter((record) => metadataMatches(record.metadata, filters))
      .map((record) => {
        const similarity = cosineSimilarity(embedding, record.embedding ?? []);
        return {
          ...record,
          similarity,
          score: Math.max(0, similarity * effectiveImportance(record)),
        };
      })
      .filter((record) => (record.similarity ?? 0) >= threshold)
      .sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
      .slice(0, options.k ?? 5);
    const ids = new Set(results.map((record) => record.id));
    for (const record of records) {
      if (ids.has(record.id)) {
        record.accessedAt = now();
        record.accessCount += 1;
      }
    }
    await this.store.save(records);
    return results.map(({ embedding: _embedding, ...record }) => record);
  }

  async inject(prompt: string, k = 5): Promise<string> {
    const memories = await this.search(prompt, k);
    if (memories.length === 0) return prompt;
    const block = memories.map((memory) => `- ${memory.text}`).join("\n");
    return `[Memory context]\n${block}\n\n${prompt}`;
  }

  async hybridSearch(query: string, options: SearchOptions = {}): Promise<MemoryRecord[]> {
    const semantic = await this.search(query, { ...options, k: Math.max((options.k ?? 5) * 2, options.k ?? 5) });
    const terms = new Set(normalizeText(query, 20_000).toLowerCase().match(/[a-z0-9][a-z0-9_'-]*/g) ?? []);
    if (terms.size === 0) return semantic.slice(0, options.k ?? 5);
    const records = this.http ? [] : await this.store.load();
    const seen = new Map(semantic.map((record) => [record.id, record]));
    for (const record of records) {
      if (record.namespace !== normalizeNamespace(options.namespace ?? this.namespace)) continue;
      if (options.memoryTypes && !options.memoryTypes.includes(record.memoryType ?? "long_term")) continue;
      const words = record.text.toLowerCase().match(/[a-z0-9][a-z0-9_'-]*/g) ?? [];
      const overlap = words.filter((word) => terms.has(word)).length;
      if (overlap === 0) continue;
      const keywordScore = overlap / terms.size;
      const existing = seen.get(record.id);
      if (existing) {
        existing.score = Math.max(existing.score ?? 0, keywordScore) + 0.15;
      } else {
        seen.set(record.id, { ...record, score: keywordScore, similarity: keywordScore });
      }
    }
    return [...seen.values()].sort((left, right) => (right.score ?? 0) - (left.score ?? 0)).slice(0, options.k ?? 5);
  }

  async learn(userMsg: string, assistantMsg: string): Promise<string[]> {
    void assistantMsg;
    const facts = extractFacts(normalizeText(userMsg, 20_000));
    for (const fact of facts) await this.save(fact, { metadata: { source: "learn" } });
    return facts;
  }

  async forget(query: string): Promise<number> {
    if (this.http) return this.http.forget(query);
    const matches = await this.search(query, { k: 5, threshold: 0.75 });
    const ids = new Set(matches.map((record) => record.id));
    const records = await this.store.load();
    const kept = records.filter((record) => !ids.has(record.id));
    await this.store.save(kept);
    return records.length - kept.length;
  }

  async clear(): Promise<void> {
    if (this.http) {
      await this.http.clear();
      return;
    }
    const records = await this.store.load();
    await this.store.save(records.filter((record) => record.namespace !== this.namespace));
  }

  async export(path: string): Promise<void> {
    const payload = this.http
      ? await this.http.exportPayload()
      : { version: 1 as const, dimension: this.dimension, records: await this.store.load() };
    if (typeof globalThis.window !== "undefined") {
      globalThis.localStorage.setItem(`memex:export:${path}`, JSON.stringify(payload));
      return;
    }
    const fs = await import("node:fs/promises");
    await fs.writeFile(path, JSON.stringify(payload, null, 2), "utf8");
  }

  async importFrom(path: string): Promise<number> {
    if (this.http) throw new Error("HTTP import is available through the Python REST /import endpoint");
    let payload: ExportPayload;
    if (typeof globalThis.window !== "undefined") {
      const raw = globalThis.localStorage.getItem(`memex:export:${path}`);
      if (!raw) throw new Error("browser export key was not found");
      payload = JSON.parse(raw) as ExportPayload;
    } else {
      const fs = await import("node:fs/promises");
      payload = JSON.parse(await fs.readFile(path, "utf8")) as ExportPayload;
    }
    const records = await this.store.load();
    const existing = new Set(records.map((record) => record.id));
    let imported = 0;
    for (const record of payload.records) {
      if (existing.has(record.id)) continue;
      records.push(record);
      imported += 1;
    }
    await this.store.save(records);
    return imported;
  }

  private dedupe(records: MemoryRecord[], inserted: MemoryRecord): void {
    const closest = records
      .filter((record) => record.id !== inserted.id && record.namespace === inserted.namespace)
      .map((record) => ({
        record,
        similarity: cosineSimilarity(inserted.embedding ?? [], record.embedding ?? []),
      }))
      .sort((left, right) => right.similarity - left.similarity)[0];
    if (closest && closest.similarity >= this.dedupeThreshold) {
      const index = records.findIndex((record) => record.id === closest.record.id);
      if (index >= 0) records.splice(index, 1);
    }
  }

  private prune(records: MemoryRecord[], namespace: string): void {
    const scoped = records.filter((record) => record.namespace === namespace);
    if (scoped.length <= this.maxMemories) return;
    const toDelete = new Set(
      scoped
        .sort((left, right) => effectiveImportance(left) - effectiveImportance(right))
        .slice(0, Math.max(scoped.length - this.maxMemories, Math.floor(scoped.length / 10)))
        .map((record) => record.id),
    );
    for (let index = records.length - 1; index >= 0; index -= 1) {
      if (toDelete.has(records[index]?.id ?? "")) records.splice(index, 1);
    }
  }
}

export type { ExportPayload, MemoryOptions, MemoryRecord, Metadata, SaveOptions, SearchOptions };
