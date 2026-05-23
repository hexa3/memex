import express from "express";
import { Memory } from "memex-ai";

const app = express();
const mem = new Memory();

app.use(express.json());

app.post("/message", async (req, res) => {
  const prompt = String(req.body.prompt ?? "");
  const enriched = await mem.inject(prompt);
  res.json({ prompt: enriched });
});

app.listen(3000, () => {
  console.log("listening on http://127.0.0.1:3000");
});
