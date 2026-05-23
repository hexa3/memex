export type Metadata = Record<string, unknown>;

export interface MemoryRecord {
  id: string;
  namespace: string;
  text: string;
  embedding?: number[] | undefined;
  metadata?: Metadata | undefined;
  importance: number;
  createdAt: number;
  accessedAt: number;
  accessCount: number;
  ttlDays?: number | undefined;
  score?: number | undefined;
  similarity?: number | undefined;
}

export interface MemoryOptions {
  namespace?: string | undefined;
  endpoint?: string | undefined;
  storagePath?: string | undefined;
  dimension?: number | undefined;
  autoDedupe?: boolean | undefined;
  dedupeThreshold?: number | undefined;
  maxMemories?: number | undefined;
}

export interface SaveOptions {
  metadata?: Metadata | undefined;
  ttlDays?: number | undefined;
  importance?: number | undefined;
  namespace?: string | undefined;
}

export interface SearchOptions {
  k?: number | undefined;
  threshold?: number | undefined;
  filters?: Metadata | undefined;
  namespace?: string | undefined;
}

export interface ExportPayload {
  version: 1;
  dimension: number;
  records: MemoryRecord[];
}
