import { Memory } from "./js/dist/index.js";

const mem = new Memory({ endpoint: "http://127.0.0.1:8765" });

await mem.save("hello from JS");
const result = await mem.recall("hello");
console.log(result);