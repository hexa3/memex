# memex-ai

TypeScript client for memex.

```ts
import { Memory } from "memex-ai";

const mem = new Memory();
await mem.save("The user prefers dark mode.");
console.log(await mem.recall("theme preference"));
```

Use a Python REST server:

```ts
const mem = new Memory({ endpoint: "http://127.0.0.1:8765" });
```
