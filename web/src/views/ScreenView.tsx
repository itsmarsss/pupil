import { C } from "../colors";
import { useEngine } from "../engine";
import { FitBox } from "./FitBox";
import type { Vec2 } from "../protocol";

/** The display, in points: calibration dots, the current target and every model's estimate. */
export function ScreenView() {
  const { scene, frame, calibration, trail } = useEngine();
  if (!scene) return <div className="placeholder">waiting for the engine</div>;
  const { width_pt: W, height_pt: H } = scene.screen;
  const g = frame?.gaze;
  const r = W / 120;
  const dot = (p: Vec2 | null | undefined, color: string, size = 1, label?: string) =>
    p && (
      <g>
        <circle cx={p[0]} cy={p[1]} r={r * size} fill={color} />
        {label && <text x={p[0] + r * 1.6} y={p[1] + r * 0.5} fill={color} fontSize={r * 2.2}>{label}</text>}
      </g>
    );
  return (
    <FitBox aspect={W / H}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        <rect x={0} y={0} width={W} height={H} fill="#151922" stroke={C.screen} strokeWidth={2} />
        {calibration?.targets?.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={r * 0.6} fill="none" stroke={C.calib} strokeWidth={2} />
        ))}
        {trail.length > 1 && (
          <polyline points={trail.map((p) => p.join(",")).join(" ")} fill="none" stroke={C.smoothed}
                    strokeOpacity={0.35} strokeWidth={3} />
        )}
        {frame?.target && g?.fused && (
          <line x1={frame.target[0]} y1={frame.target[1]} x2={g.fused[0]} y2={g.fused[1]}
                stroke={C.fused} strokeDasharray="6 6" strokeWidth={2} />
        )}
        {frame?.target && (
          <circle cx={frame.target[0]} cy={frame.target[1]} r={r * 1.8} fill="none" stroke={C.target} strokeWidth={3} />
        )}
        {dot(frame?.face_pointing_pt, C.pointing, 0.8, "face")}
        {dot(g?.ml, C.ml, 0.9, "ML")}
        {dot(g?.eyeball, C.eyeball, 0.9, "eyeball")}
        {dot(g?.fused, C.fused, 1.1)}
        {g?.smoothed && (
          <circle cx={g.smoothed[0]} cy={g.smoothed[1]} r={r * 2.4} fill="none" stroke={C.smoothed} strokeWidth={3} />
        )}
      </svg>
    </FitBox>
  );
}
