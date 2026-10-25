// Messages from the engine (src/gaze/telemetry.py, schema 1).
// Image coordinates are normalised 0..1; screen coordinates are points (top-left origin);
// 3D is camera space in cm: +x = image right (the user's left), +y up, +z toward the user's back.

export const SCHEMA = 1;

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface SceneInfo {
  type: "scene";
  schema: number;
  screen: { width_pt: number; height_pt: number; width_mm: number; height_mm: number; corners_cm: Vec3[] };
  camera: { focal_px_1080: number; measured: boolean; hfov_deg: number };
  lid_deg: number | null;
  replay?: { session: string; frames: number; markers: { index: number; phase: string | null }[] };
}

export interface CalibrationInfo {
  type: "calibration";
  schema: number;
  calibrated: boolean;
  profile?: string;
  chosen?: string;
  cv_error_pt?: Record<string, number | null>;
  dots?: number;
  head_motion?: boolean;
  clicks?: number;
  eyeball?: { gain: Vec2; kappa_deg: Vec2; residual: boolean; head_comp: boolean } | null;
  lid_deg?: number | null;
  offset_pt?: Vec2;
  targets?: Vec2[];
}

export interface Gaze {
  fused: Vec2 | null;
  smoothed: Vec2 | null;
  fixation: number | null;
  ml?: Vec2;
  eyeball?: Vec2 | null;
  disagreement_pt?: number | null;
}

export interface FrameScene {
  points?: Vec3[];
  eyeballs?: [Vec3, Vec3];
  eyeball_radius?: number;
  iris?: [Vec3, Vec3];
  head?: { origin: Vec3; axes: [Vec3, Vec3, Vec3]; nose: Vec3 };
  eye_angles?: [Vec2, Vec2];
  face_pointing?: Vec3 | null;
  rays?: { dirs: [Vec3, Vec3]; hits: [Vec3, Vec3] | null };
  hits?: Partial<Record<"ml" | "eyeball" | "fused", Vec3>>;
}

export interface FrameState {
  type: "frame";
  schema: number;
  t: number;
  latency_ms: number;
  fps: number;
  state: "tracking" | "no_face" | "blink" | "uncalibrated";
  confidence: number | null;
  distance_cm: number | null;
  gaze: Gaze;
  image?: {
    size: Vec2;
    landmarks: Vec2[];
    iris: [Vec2, Vec2];
    boxes: Record<"face" | "left_eye" | "right_eye", [number, number, number, number]>;
  };
  eyes_open?: Vec2;
  mouth_open?: number | null; // inner-lip gap / mouth width
  head_ypr?: Vec3;
  face_pointing_pt?: Vec2 | null;
  scene?: FrameScene;
  // from the app
  phase?: string;
  target?: Vec2 | null;
  busy?: boolean;
  lid_deg?: number | null;
  smoothing?: string;
  overlay?: boolean;
  replay?: { index: number; playing: boolean; speed: number };
  voice?: VoiceState | null;
}

/** What the voice controller (gaze control) heard, decided and is doing. */
export interface VoiceState {
  connected: boolean;
  mic?: string | null;
  heard?: string;
  pending?: string | null;
  target?: { x: number; y: number; w: number; h: number; label: string } | null;
  log: { t: number; kind: "heard" | "status" | string; text: string }[];
  lip_gate?: LipGateState;
  lips?: LipVerdict[];
}

/** The live lip gate (control/lips.py): audio only counts while it's open. */
export interface LipGateState {
  mode: "on" | "shadow";
  open: boolean;
  reason: string; // open | lips still | not looking | no tracker - open
  threshold: number; // lip movement (max - min over 0.4 s) that opens it; adapts to your resting jitter
  open_above: number; // or a mouth held open wider than this
}

/** The gate's verdict on one transcript; t0/t1 on the frame clock (FrameState.t). */
export interface LipVerdict {
  t0: number;
  t1: number;
  moving: boolean; // false: your face was visible and your lips never moved, so it was dropped
  why?: string;
}

export type EngineMessage = SceneInfo | CalibrationInfo | FrameState;

export type Command =
  | "recalibrate" | "recentre" | "check" | "measure" | "cycle_smoothing" | "toggle_overlay" // live app
  | "play" | "pause" | "seek" | "step" | "speed"; // replay
