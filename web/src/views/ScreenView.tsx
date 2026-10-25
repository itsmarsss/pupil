import { C } from "../colors";
import { useEngine } from "../engine";
import { FitBox } from "./FitBox";
import type { Vec2 } from "../protocol";

const HEAT_POINTS = 30; // recent gaze points drawn as heat (the trail keeps ~1.5 s)

/** The display, in points: calibration dots, the current target, every model's estimate, and gaze as a heat map
 *  (recent points blurred by the tracker's measured accuracy), with the last "this"/"that" candidates' shares. */
export function ScreenView() {
  const { scene, frame, calibration, trail } = useEngine();
  if (!scene) return <div className="pane placeholder">waiting for the engine</div>;
  const { width_pt: W, height_pt: H } = scene.screen;
  const g = frame?.gaze;
  const r = W / 120;
  const sigma = frame?.gaze_sigma_pt;
  const dot = (p: Vec2 | null | undefined, color: string, size = 1, label?: string) =>
    p && (
      <g>
        <circle cx={p[0]} cy={p[1]} r={r * size} fill={color} />
        {label && <text x={p[0] + r * 1.6} y={p[1] + r * 0.5} fill={color} fontSize={r * 2.2}>{label}</text>}
      </g>
    );
  return (
    <div className="pane">
    <FitBox aspect={W / H}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        <defs>
          {/* a Gaussian falloff: each recent gaze point adds one of these, so overlapping ones read as heat */}
          <radialGradient id="gaze-heat">
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <stop key={t} offset={t} stopColor="#ff6b3d" stopOpacity={Math.exp(-((t * 2.5) ** 2) / 2)} />
            ))}
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="#151922" stroke={C.screen} strokeWidth={2} />
        {sigma && trail.slice(-HEAT_POINTS).map((p, i) => (
          <circle key={`h${i}`} cx={p[0]} cy={p[1]} r={sigma * 2.5} fill="url(#gaze-heat)" opacity={0.1} />
        ))}
        {frame?.voice?.heat?.map((o, i) => (
          <g key={`c${i}`}>
            <rect x={o.x} y={o.y} width={o.w} height={o.h} fill="none" stroke="#ffb23d" strokeWidth={2}
                  strokeOpacity={0.3 + 0.7 * o.share} rx={4} />
            <text x={o.x + 4} y={o.y + o.h + r * 2.4} fill="#ffb23d" fontSize={r * 2}>
              {`${Math.round(o.share * 100)}% ${o.label.slice(0, 18)}`}
            </text>
          </g>
        ))}
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
        {frame?.voice?.target && (
          <rect x={frame.voice.target.x} y={frame.voice.target.y} width={frame.voice.target.w}
                height={frame.voice.target.h} fill="none" stroke={C.voice} strokeWidth={3} rx={4} />
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
    </div>
  );
}
