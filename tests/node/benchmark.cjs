#!/usr/bin/env node

/* Same-runner, warmed full-OCR comparison for canonical vs compiled WASM. */
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
  const modelsPath = argument("--models");
  const expectedSha = argument("--expected-text-sha256");
  const iterations = Number(argument("--iterations", "5"));
  if (!packagePath || !samplePath ||
      !Number.isInteger(iterations) || iterations < 1 || iterations > 100) {
    throw new Error("usage: benchmark.cjs --package <directory> --sample <P6 ppm> " +
                    "[--models <verified runtime-assets directory>] " +
                    "[--expected-text-sha256 <digest>] [--iterations 5]");
  }
  const root = path.resolve(packagePath);
  const modelRoot = modelsPath ? path.resolve(modelsPath) : root;
  const sample = path.resolve(samplePath);
  const image = readPpm(sample);
  const sampleSha = crypto.createHash("sha256").update(fs.readFileSync(sample)).digest("hex");
  const manifest = verifyPackage(root);
  let modelVariant = "tiny";
  if (modelsPath) {
    const assetManifest = JSON.parse(fs.readFileSync(
      path.join(modelRoot, "runtime-assets.json"), "utf8"));
    if (assetManifest.schema_version !== 1 ||
        !["small", "medium"].includes(assetManifest.variant)) {
      throw new Error("unsupported runtime model asset manifest");
    }
    for (const [name, digest] of Object.entries(assetManifest.assets)) {
      const actual = crypto.createHash("sha256").update(
        fs.readFileSync(path.join(modelRoot, name))).digest("hex");
      if (actual !== digest) throw new Error(`model asset checksum mismatch: ${name}`);
    }
    modelVariant = assetManifest.variant;
  }
  // The checked-in Tiny Web/Node golden is captured with CLS enabled.
  const engine = await createRuntime(root, true, modelRoot);
  let peakRss = process.memoryUsage().rss;
  try {
    let lines = runOcr(engine, image); // warm the physical program and heap
    const expectedText = lines.map(line => line.text).join("\n");
    const expectedDetectedCount = lines.detectedCount;
    const digest = crypto.createHash("sha256").update(expectedText, "utf8").digest("hex");
    if (expectedSha && digest !== expectedSha) {
      throw new Error(`OCR checksum mismatch: expected ${expectedSha}, actual ${digest}; ` +
        `lines: ${JSON.stringify(lines.map(line => line.text))}`);
    }
    const timings = [];
    for (let index = 0; index < iterations; index++) {
      const started = process.hrtime.bigint();
      lines = runOcr(engine, image);
      const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
      if (lines.map(line => line.text).join("\n") !== expectedText) {
        throw new Error(`OCR text changed on measured iteration ${index}`);
      }
      if (lines.detectedCount !== expectedDetectedCount) {
        throw new Error(`DET box count changed on measured iteration ${index}`);
      }
      timings.push(elapsed);
      peakRss = Math.max(peakRss, process.memoryUsage().rss);
    }
    timings.sort((a, b) => a - b);
    console.log(JSON.stringify({
      schema_version: 1,
      wasm_backend: manifest.runtime.backend,
      model_variant: modelVariant,
      use_cls: true,
      sample_sha256: sampleSha,
      iterations,
      median_ms: timings[Math.floor(timings.length / 2)],
      min_ms: timings[0],
      max_ms: timings[timings.length - 1],
      peak_rss_bytes: Math.max(peakRss, process.resourceUsage().maxRSS * 1024),
      wasm_heap_bytes: engine.runtime.HEAPU8.byteLength,
      line_count: lines.length,
      detected_count: lines.detectedCount,
      text_sha256: digest,
      text_lines: lines.map(line => line.text)
    }));
  } finally {
    engine.runtime._lw_web_shutdown();
  }
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
