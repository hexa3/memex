import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { Memory } from "./index.js";

test("save and recall", async () => {
  const dir = await mkdtemp(join(tmpdir(), "memex-js-"));
  try {
    const mem = new Memory({ storagePath: join(dir, "memex.json") });
    await mem.save("User prefers dark mode");
    assert.equal(await mem.recall("dark mode"), "User prefers dark mode");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("learn extracts durable facts", async () => {
  const dir = await mkdtemp(join(tmpdir(), "memex-js-"));
  try {
    const mem = new Memory({ storagePath: join(dir, "memex.json") });
    const facts = await mem.learn("My name is Jules. I prefer concise answers.", "Noted.");
    assert.ok(facts.includes("User's name is Jules."));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
