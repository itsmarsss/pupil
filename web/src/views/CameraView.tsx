import { C } from "../colors";
import { useEngine } from "../engine";

/** Camera image with landmarks, iris centres and the crops the gaze model sees. */
export function CameraView() {
  const { imageUrl, frame } = useEngine();
  const img = frame?.image;
  const aspect = img ? img.size[0] / img.size[1] : 4 / 3;
  return (
    <div className="media" style={{ aspectRatio: aspect }}>
      {imageUrl ? <img src={imageUrl} alt="camera" /> : <div className="placeholder">no image</div>}
      {img && (
        <svg viewBox={`0 0 ${aspect} 1`} preserveAspectRatio="none">
          <g transform={`scale(${aspect} 1)`}>
            {img.landmarks.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={0.0018} fill={C.face} />
            ))}
            {img.iris.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={0.005} fill={C.iris} />
            ))}
            {Object.entries(img.boxes).map(([name, [x, y, w, h]]) => (
              <rect key={name} x={x} y={y} width={w} height={h} fill="none" stroke={C.boxes}
                    strokeWidth={0.002} strokeDasharray={name === "face" ? "0.01 0.006" : undefined} />
            ))}
          </g>
        </svg>
      )}
    </div>
  );
}
