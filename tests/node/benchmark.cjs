#!/usr/bin/env node

/* Same-process, warmed full-OCR comparison for canonical vs compiled WASM. */
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const {createRuntime, readPpm, runOcr, verifyPackage} = require("./smoke.cjs");

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

async function main() {
  const packagePath = argument("--package");
  const samplePath = argument("--sample");
  const expectedSha = argument("--expected-text-sha256");
  const iterations = Number(argument("--iterations", "5"));
  if (!packagePath || !samplePath || !expectedSha ||
      !Number.isInteger(iterations) || iterations < 1 || iterations > 100) {
    throw new Error("usage: benchmark.cjs --package <directory> --sample <P6 ppm> " +
                    "--expected-text-sha256 <digest> [--iterations 5]");
  }
  const root = path.resolve(packagePath);
  const sample = path.resolve(samplePath);
  const image = readPpm(sample);
  const sampleSha = crypto.createHash("sha256").update(fs.readFileSync(sample)).digest("hex");
  const manifest = verifyPackage(root);
  const engine = await createRuntime(root, false);
  let peakRss = process.memoryUsage().rss;
  try {
    let lines = runOcr(engine, image); // warm the physical program and heap
    const expectedText = lines.map(line => line.text).join("\n");
    const digest = crypto.createHash("sha256").update(expectedText, "utf8").digest("hex");
    if (digest !== expectedSha) throw new Error(`OCR checksum mismatch: ${digest}`);
    const timings = [];
    for (let index = 0; index < iterations; index++) {
      const started = process.hrtime.bigint();
      lines = runOcr(engine, image);
      const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
      if (lines.map(line => line.text).join("\n") !== expectedText) {
        throw new Error(`OCR text changed on measured iteration ${index}`);
      }
      timings.push(elapsed);
      peakRss = Math.max(peakRss, process.memoryUsage().rss);
    }
    timings.sort((a, b) => a - b);
    console.log(JSON.stringify({
      schema_version: 1,
      wasm_backend: manifest.runtime.backend,
      sample_sha256: sampleSha,
      iterations,
      median_ms: timings[Math.floor(timings.length / 2)],
      min_ms: timings[0],
      max_ms: timings[timings.length - 1],
      peak_rss_bytes: Math.max(peakRss, process.resourceUsage().maxRSS * 1024),
      wasm_heap_bytes: engine.runtime.HEAPU8.byteLength,
      line_count: lines.length,
      text_sha256: digest
    }));
  } finally {
    engine.runtime._lw_web_shutdown();
  }
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
