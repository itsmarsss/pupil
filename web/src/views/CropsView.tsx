import { useEngine } from "../engine";

/** What the gaze network actually receives: the 224 px face crop and the two 112 px eye crops, after the
 * eyes are levelled. The right eye is mirrored, as the network expects. */
export function CropsView() {
  const { cropsUrl } = useEngine();
  return (
    <div className="crops">
      <div className="crops-label">Model input <span>face · left eye · right eye (mirrored)</span></div>
      {cropsUrl ? <img src={cropsUrl} alt="model input crops" /> : <div className="placeholder">no crops yet</div>}
    </div>
  );
}
