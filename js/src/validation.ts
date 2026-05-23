import type { Metadata } from "./types.js";

const namespacePattern = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

export function normalizeNamespace(namespace?: string): string {
  const value = (namespace ?? "default").trim();
  if (!namespacePattern.test(value)) {
    throw new Error("namespace must be 1-128 safe characters");
  }
  return value;
}

export function normalizeText(text: string, maxChars = 100_000): string {
  if (typeof text !== "string") throw new Error("text must be a string");
  const value = text.trim();
  if (!value) throw new Error("text must not be empty");
  if (value.length > maxChars) throw new Error(`text exceeds ${maxChars} characters`);
  return value;
}

export function validateMetadata(metadata?: Metadata): Metadata | undefined {
  if (metadata === undefined) return undefined;
  if (metadata === null || Array.isArray(metadata) || typeof metadata !== "object") {
    throw new Error("metadata must be an object");
  }
  const encoded = JSON.stringify(metadata);
  if (encoded.length > 65_536) throw new Error("metadata exceeds 65536 bytes");
  return metadata;
}

export function metadataMatches(metadata: Metadata | undefined, filters: Metadata | undefined): boolean {
  if (!filters) return true;
  if (!metadata) return false;
  return Object.entries(filters).every(([key, value]) => metadata[key] === value);
}
