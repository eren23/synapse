// Web Worker: loads UniXcoder + CDT in an off-main-thread context.
//
// Message protocol:
//   { type: 'load', unixcoderBytes: Uint8Array, cdtBytes?: Uint8Array }
//     → { type: 'loaded', hiddenSize, cdtK?, cdtD? }
//     or { type: 'error', message }
//   { type: 'encode', idsBefore, maskBefore, idsAfter, maskAfter }
//     → { type: 'encoded',
//         hb: Float32Array, ha: Float32Array,
//         delta?: Float32Array, recon?: Float32Array,
//         timingMs: { cls, cdt } }
//
// The worker's stdlib fetch for the wasm file is relative to *this* JS
// file's URL, hence the ../synapse-wasm-pkg path.

import init, { WasmUnixcoder, WasmCodeDeltaTok } from '../synapse-wasm-pkg/synapse_wasm.js';

// Forward the Worker's console output to the main thread so panic
// messages from console_error_panic_hook become visible in the demo's
// log pane (Workers have their own console that the main page cannot
// read otherwise).
for (const level of ['log', 'warn', 'error']) {
  const orig = console[level].bind(console);
  console[level] = (...args) => {
    try {
      postMessage({ type: 'log', level, text: args.map(String).join(' ') });
    } catch {}
    orig(...args);
  };
}

let unixcoder = null;
let cdt = null;

async function ensureInit() {
  await init();
}

async function handleLoad(msg) {
  await ensureInit();
  unixcoder = new WasmUnixcoder(msg.unixcoderBytes);
  const payload = { type: 'loaded', hiddenSize: unixcoder.hidden_size() };
  if (msg.cdtBytes && msg.cdtBytes.byteLength > 0) {
    cdt = new WasmCodeDeltaTok(msg.cdtBytes);
    payload.cdtK = cdt.num_delta_tokens();
    payload.cdtD = cdt.feature_dim();
  }
  postMessage(payload);
}

function handleEncode(msg) {
  if (!unixcoder) throw new Error('UniXcoder not loaded');
  const t0 = performance.now();
  const hb = unixcoder.cls_feature(msg.idsBefore, msg.maskBefore);
  const ha = unixcoder.cls_feature(msg.idsAfter,  msg.maskAfter);
  const t1 = performance.now();

  let delta, recon;
  if (cdt) {
    delta = cdt.encode(hb, ha);
    recon = cdt.decode(delta, hb);
  }
  const t2 = performance.now();

  postMessage(
    {
      type: 'encoded',
      hb, ha, delta, recon,
      timingMs: { cls: t1 - t0, cdt: t2 - t1 },
    },
    // Transfer the heavy buffers to avoid a copy back.
    [hb.buffer, ha.buffer,
     ...(delta ? [delta.buffer] : []),
     ...(recon ? [recon.buffer] : [])],
  );
}

onmessage = async (ev) => {
  try {
    const msg = ev.data;
    if (msg.type === 'load') await handleLoad(msg);
    else if (msg.type === 'encode') handleEncode(msg);
    else throw new Error(`Unknown message type: ${msg.type}`);
  } catch (e) {
    postMessage({ type: 'error', message: String(e) });
  }
};
