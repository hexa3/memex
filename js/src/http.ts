import type { ExportPayload, MemoryRecord, SaveOptions, SearchOptions } from "./types.js";

function toRecord(raw: Record<string, unknown>): MemoryRecord {
  return {
    id: String(raw.id),
    namespace: String(raw.namespace ?? "default"),
    text: String(raw.text),
    metadata: raw.metadata as Record<string, unknown> | undefined,
    importance: Number(raw.importance ?? 1),
    createdAt: Date.parse(String(raw.created_at ?? raw.createdAt ?? new Date().toISOString())) / 1000,
    accessedAt: Date.parse(String(raw.accessed_at ?? raw.accessedAt ?? new Date().toISOString())) / 1000,
    accessCount: Number(raw.access_count ?? raw.accessCount ?? 0),
    ttlDays: raw.ttl_days === undefined ? undefined : Number(raw.ttl_days),
    score: raw.score === undefined ? undefined : Number(raw.score),
    similarity: raw.similarity === undefined ? undefined : Number(raw.similarity),
  };
}

export class HttpClient {
  private readonly endpoint: string;

  constructor(endpoint: string) {
    this.endpoint = endpoint.replace(/\/+$/, "");
  }

  async save(text: string, options: SaveOptions = {}): Promise<string> {
    const response = await fetch(`${this.endpoint}/save`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text,
        metadata: options.metadata,
        ttl_days: options.ttlDays,
        importance: options.importance,
        namespace: options.namespace,
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    const body = (await response.json()) as { id: string };
    return body.id;
  }

  async search(query: string, options: SearchOptions = {}): Promise<MemoryRecord[]> {
    const params = new URLSearchParams({
      q: query,
      k: String(options.k ?? 5),
      threshold: String(options.threshold ?? 0),
    });
    if (options.namespace) params.set("namespace", options.namespace);
    const response = await fetch(`${this.endpoint}/search?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    const body = (await response.json()) as { results: Record<string, unknown>[] };
    return body.results.map(toRecord);
  }

  async recall(query: string): Promise<string | null> {
    const params = new URLSearchParams({ q: query });
    const response = await fetch(`${this.endpoint}/recall?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    const body = (await response.json()) as { text: string | null };
    return body.text;
  }

  async learn(userMsg: string, assistantMsg: string): Promise<string[]> {
    const response = await fetch(`${this.endpoint}/learn`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ user_msg: userMsg, assistant_msg: assistantMsg }),
    });
    if (!response.ok) throw new Error(await response.text());
    const body = (await response.json()) as { saved: string[] };
    return body.saved;
  }

  async forget(query: string): Promise<number> {
    const params = new URLSearchParams({ q: query });
    const response = await fetch(`${this.endpoint}/forget?${params.toString()}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await response.text());
    const body = (await response.json()) as { deleted: number };
    return body.deleted;
  }

  async clear(): Promise<void> {
    const response = await fetch(`${this.endpoint}/clear`, { method: "DELETE" });
    if (!response.ok) throw new Error(await response.text());
  }

  async exportPayload(): Promise<ExportPayload> {
    const response = await fetch(`${this.endpoint}/export`);
    if (!response.ok) throw new Error(await response.text());
    return (await response.json()) as ExportPayload;
  }
}
