"use strict";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

async function sha256(data) {
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function inflateRaw(data) {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function readZip(buffer) {
  const bytes = new Uint8Array(buffer); const view = new DataView(buffer); const files = new Map();
  let end = -1;
  for (let index = bytes.length - 22; index >= Math.max(0, bytes.length - 65557); index -= 1) {
    if (view.getUint32(index, true) === 0x06054b50) { end = index; break; }
  }
  if (end < 0) throw new Error("Not a ZIP/PFCASE container.");
  const entries = view.getUint16(end + 10, true); let cursor = view.getUint32(end + 16, true);
  for (let count = 0; count < entries; count += 1) {
    if (view.getUint32(cursor, true) !== 0x02014b50) throw new Error("Invalid ZIP directory.");
    const method = view.getUint16(cursor + 10, true); const compressedSize = view.getUint32(cursor + 20, true);
    const nameLength = view.getUint16(cursor + 28, true); const extraLength = view.getUint16(cursor + 30, true); const commentLength = view.getUint16(cursor + 32, true); const localOffset = view.getUint32(cursor + 42, true);
    const name = decoder.decode(bytes.slice(cursor + 46, cursor + 46 + nameLength));
    if (name.includes("..") || name.startsWith("/") || files.has(name)) throw new Error(`Unsafe ZIP entry: ${name}`);
    const localNameLength = view.getUint16(localOffset + 26, true); const localExtraLength = view.getUint16(localOffset + 28, true); const start = localOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.slice(start, start + compressedSize); const content = method === 0 ? compressed : method === 8 ? await inflateRaw(compressed) : null;
    if (!content) throw new Error(`Unsupported ZIP compression method ${method}.`); files.set(name, content);
    cursor += 46 + nameLength + extraLength + commentLength;
  }
  return files;
}

async function merkle(leaves) {
  let layer = [...leaves]; if (!layer.length) return sha256(new Uint8Array());
  while (layer.length > 1) { if (layer.length % 2) layer.push(layer.at(-1)); const next = []; for (let index = 0; index < layer.length; index += 2) next.push(await sha256(encoder.encode(layer[index] + layer[index + 1]))); layer = next; }
  return layer[0];
}

function addCheck(label, valid, detail) {
  const row = document.createElement("article"); row.className = `check${valid ? "" : " fail"}`;
  const name = document.createElement("span"); name.textContent = label; const state = document.createElement("strong"); state.textContent = valid ? "VALID" : detail || "FAILED"; row.append(name, state); document.querySelector("#checks").append(row);
}

async function verify(file) {
  document.querySelector("#error").textContent = ""; document.querySelector("#checks").replaceChildren();
  const files = await readZip(await file.arrayBuffer()); if (!files.has("manifest.json")) throw new Error("manifest.json is missing.");
  const manifest = JSON.parse(decoder.decode(files.get("manifest.json"))); document.querySelector("#results").hidden = false; document.querySelector("#case-id").textContent = manifest.case_id || "UNKNOWN";
  const names = Object.keys(manifest.files || {}).sort(); let hashesValid = true; let missing = 0;
  for (const name of names) { const content = files.get(name); if (!content) { missing += 1; hashesValid = false; continue; } if (await sha256(content) !== manifest.files[name]) hashesValid = false; }
  const root = await merkle(names.map((name) => manifest.files[name])); const rootValid = root === manifest.merkle_root;
  const required = ["case.json","evidence/events.jsonl","evidence/sources.json","commitments/epochs.json","contracts/evidence-contract.json","verification.json"];
  const schemaValid = manifest.bundle_version === "PFCASE-1.0" && required.every((name) => files.has(name));
  addCheck("PFCASE schema", schemaValid, "INVALID"); addCheck("Declared files", missing === 0, `${missing} MISSING`); addCheck("Evidence hashes", hashesValid, "MODIFIED"); addCheck("Merkle commitment", rootValid, "INVALID"); addCheck("Manifest signature", Boolean(manifest.manifest_signature), "NOT SUPPLIED"); addCheck("External checkpoint", Boolean(manifest.external_anchor), "NOT SUPPLIED");
  const contract = files.has("contracts/evidence-contract.json") ? JSON.parse(decoder.decode(files.get("contracts/evidence-contract.json"))) : {}; const sources = files.has("evidence/sources.json") ? JSON.parse(decoder.decode(files.get("evidence/sources.json"))) : [];
  const observed = new Set(sources.map((item) => typeof item === "string" ? item : item.producer_id || item.source)); const missingSources = (contract.expected_sources || []).filter((source) => !observed.has(source)); document.querySelector("#missing-sources").textContent = missingSources.join(" / ") || "NONE DECLARED"; document.querySelector("#unknown-risk").textContent = manifest.unknown_omission_risk || "NOT ELIMINATED";
  const valid = schemaValid && hashesValid && rootValid && missing === 0; document.querySelector("#case-state").textContent = valid ? "SUPPLIED RECORD VERIFIED" : "VERIFICATION FAILED";
}

const input = document.querySelector("#bundle"); const drop = document.querySelector("#drop-zone");
input.addEventListener("change", () => input.files[0] && verify(input.files[0]).catch((error) => document.querySelector("#error").textContent = error.message));
["dragenter","dragover"].forEach((event) => drop.addEventListener(event, (value) => { value.preventDefault(); drop.classList.add("drag"); }));
["dragleave","drop"].forEach((event) => drop.addEventListener(event, (value) => { value.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) verify(file).catch((error) => document.querySelector("#error").textContent = error.message); });
