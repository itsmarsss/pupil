import { useEffect, useSyncExternalStore } from "react";
import type { CalibrationInfo, Command, EngineMessage, FrameState, SceneInfo } from "./protocol";
import { SCHEMA } from "./protocol";

// Live data lives in a mutable store: the 3D view reads it every animation frame, panels subscribe at a
// throttled rate. Keeping 30 Hz updates out of React state keeps the page smooth.

const HISTORY_S = 20;

// uPlot draws a gap at null (NaN would break its auto-scaling)
export interface Series {
  t: number[];
  [key: string]: (number | null)[];
}

export const SERIES_KEYS = [
  "yaw", "pitch", "roll", "distance", "open_l", "open_r", "confidence", "disagreement", "fps", "latency",
] as const;

class EngineStore {
  connected = false;
  schemaMismatch = false;
  scene: SceneInfo | null = null;
  calibration: CalibrationInfo | null = null;
  frame: FrameState | null = null;
  imageUrl: string | null = null;
  trail: [number, number][] = [];
  series: Series = { t: [], ...Object.fromEntries(SERIES_KEYS.map((k) => [k, [] as (number | null)[]])) };
  version = 0; // bumped on every update; panels poll it

  private ws: WebSocket | null = null;
  private listeners = new Set<() => void>();
  private retry: number | undefined;

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.connected = true;
      this.bump();
    };
    ws.onclose = () => {
      this.connected = false;
      this.bump();
      this.retry = window.setTimeout(() => this.connect(), 1000);
    };
    ws.onmessage = (ev) => this.receive(ev.data);
    this.ws = ws;
  }

  disconnect() {
    window.clearTimeout(this.retry);
    this.ws?.close();
  }

  send(cmd: Command) {
    this.ws?.send(JSON.stringify({ cmd }));
  }

  private receive(data: string | ArrayBuffer) {
    if (data instanceof ArrayBuffer) {
      const head = new TextDecoder().decode(data.slice(0, 4));
      if (head === "JPG0") {
        const url = URL.createObjectURL(new Blob([data.slice(4)], { type: "image/jpeg" }));
        if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
        this.imageUrl = url;
      }
      return;
    }
    const msg = JSON.parse(data) as EngineMessage;
    if (msg.schema !== SCHEMA) {
      this.schemaMismatch = true;
      return;
    }
    if (msg.type === "scene") this.scene = msg;
    else if (msg.type === "calibration") this.calibration = msg;
    else this.addFrame(msg);
    this.version++;
  }

  private addFrame(f: FrameState) {
    this.frame = f;
    if (f.gaze.smoothed) {
      this.trail.push(f.gaze.smoothed);
      if (this.trail.length > 45) this.trail.shift();
    }
    const s = this.series;
    const push = (k: string, v: number | null | undefined) => s[k].push(v ?? null);
    s.t.push(f.t);
    push("yaw", f.head_ypr?.[0]);
    push("pitch", f.head_ypr?.[1]);
    push("roll", f.head_ypr?.[2]);
    push("distance", f.distance_cm);
    push("open_l", f.eyes_open?.[0]);
    push("open_r", f.eyes_open?.[1]);
    push("confidence", f.confidence);
    push("disagreement", f.gaze.disagreement_pt);
    push("fps", f.fps);
    push("latency", f.latency_ms);
    while (s.t.length && s.t[s.t.length - 1] - s.t[0] > HISTORY_S) {
      for (const k of Object.keys(s)) s[k].shift();
    }
  }

  private bump() {
    this.version++;
    this.listeners.forEach((l) => l());
  }

  subscribe = (l: () => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  /** Notify subscribers at most `hz` times a second. */
  startTicker(hz: number) {
    let seen = -1;
    const id = window.setInterval(() => {
      if (this.version !== seen) {
        seen = this.version;
        this.listeners.forEach((l) => l());
      }
    }, 1000 / hz);
    return () => window.clearInterval(id);
  }
}

export const engine = new EngineStore();

export function useEngineConnection() {
  useEffect(() => {
    engine.connect();
    const stop = engine.startTicker(10);
    return () => {
      stop();
      engine.disconnect();
    };
  }, []);
}

/** Re-render with the store at the ticker rate (~10 Hz). */
export function useEngine() {
  useSyncExternalStore(engine.subscribe, () => engine.version);
  return engine;
}
