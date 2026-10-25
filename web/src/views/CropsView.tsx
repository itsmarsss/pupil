import { useEngine } from "../engine";
import { FitBox } from "./FitBox";

/** What the gaze network actually receives: the 224 px face crop and the two 112 px eye crops, after the
 * eyes are levelled. The right eye is mirrored, as the network expects. */
export function CropsView() {
  const { cropsUrl } = useEngine();
  return (
    <div className="pane">
      <div className="caption">face · left eye · right eye (mirrored), after levelling the eyes</div>
      {cropsUrl
        ? <FitBox aspect={336 / 224}><img src={cropsUrl} alt="model input crops" /></FitBox>
        : <div className="placeholder">no crops yet</div>}
    </div>
  );
}
