import { useEngineConnection } from "./engine";
import { CameraView } from "./views/CameraView";
import { CropsView } from "./views/CropsView";
import { Panel } from "./views/Panel";
import { Plots } from "./views/Plots";
import { Scene3D } from "./views/Scene3D";
import { Timeline } from "./views/Timeline";
import { ScreenView } from "./views/ScreenView";

export function App() {
  useEngineConnection();
  return (
    <div className="app">
      <header>Gaze dashboard<Timeline /></header>
      <main>
        <div className="scene card"><h2>3D</h2><Scene3D /></div>
        <div className="card"><h2>Camera</h2><CameraView /><CropsView /></div>
        <div className="card"><h2>Screen</h2><ScreenView /></div>
        <Panel />
      </main>
      <Plots />
    </div>
  );
}
