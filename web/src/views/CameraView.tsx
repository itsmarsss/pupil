import { useState } from "react";
import { C } from "../colors";
import { useEngine } from "../engine";
import { FitBox } from "./FitBox";

const MIRROR_KEY = "gaze.camera.mirror";

function initialMirror(): boolean {
  try {
    return localStorage.getItem(MIRROR_KEY) !== "0";
  } catch {
    return true;
  }
}

/** Camera image with landmarks, iris centres and the crops the gaze model sees. Mirrored by default, like a
 * selfie view; the image and its overlay flip together so they stay aligned. */
export function CameraView() {
  const { imageUrl, frame } = useEngine();
  const [mirror, setMirror] = useState(initialMirror);
  const img = frame?.image;
  const aspect = img ? img.size[0] / img.size[1] : 4 / 3;

  const toggle = () => {
    setMirror((m) => {
      try {
        localStorage.setItem(MIRROR_KEY, m ? "0" : "1");
      } catch {
        // storage unavailable: the choice just won't be remembered
      }
      return !m;
    });
  };

  return (
    <>
      <button className="corner-toggle" onClick={toggle}>{mirror ? "Mirrored" : "Not mirrored"}</button>
      <FitBox aspect={aspect} style={{ transform: mirror ? "scaleX(-1)" : undefined }}>
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
      </FitBox>
    </>
  );
}
