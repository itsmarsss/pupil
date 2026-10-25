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
}

export type EngineMessage = SceneInfo | CalibrationInfo | FrameState;

export type Command = "recalibrate" | "recentre" | "check" | "measure" | "cycle_smoothing" | "toggle_overlay";
