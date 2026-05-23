import type { ExportPayload, MemoryRecord } from "./types.js";

export interface Store {
  load(): Promise<MemoryRecord[]>;
  save(records: MemoryRecord[]): Promise<void>;
}

function isBrowser(): boolean {
  return typeof globalThis.window !== "undefined" && typeof globalThis.localStorage !== "undefined";
}

export class LocalStore implements Store {
  private readonly path: string | undefined;
  private readonly key: string;

  constructor(path?: string) {
    this.path = path;
    this.key = `memex:${path ?? "default"}`;
  }

  async load(): Promise<MemoryRecord[]> {
    if (isBrowser()) {
      const raw = globalThis.localStorage.getItem(this.key);
      if (!raw) return [];
      return JSON.parse(raw) as MemoryRecord[];
    }
    const fs = await import("node:fs/promises");
    const path = await this.resolvePath();
    try {
      const raw = await fs.readFile(path, "utf8");
      const parsed = JSON.parse(raw) as ExportPayload;
      return Array.isArray(parsed.records) ? parsed.records : [];
    } catch (error) {
      if (
        typeof error === "object" &&
        error !== null &&
        "code" in error &&
        (error as { code?: string }).code === "ENOENT"
      ) {
        return [];
      }
      throw error;
    }
  }

  async save(records: MemoryRecord[]): Promise<void> {
    if (isBrowser()) {
      globalThis.localStorage.setItem(this.key, JSON.stringify(records));
      return;
    }
    const fs = await import("node:fs/promises");
    const pathModule = await import("node:path");
    const path = await this.resolvePath();
    await fs.mkdir(pathModule.dirname(path), { recursive: true });
    await fs.writeFile(
      path,
      JSON.stringify({ version: 1, dimension: 384, records }, null, 2),
      { encoding: "utf8", mode: 0o600 },
    );
  }

  private async resolvePath(): Promise<string> {
    if (this.path) return this.path;
    const os = await import("node:os");
    const pathModule = await import("node:path");
    return pathModule.join(os.homedir(), ".memex", "memex-js.json");
  }
}
