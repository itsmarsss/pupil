import { DockviewReact, themeDark, type DockviewApi, type DockviewReadyEvent, type IDockviewPanelProps } from "dockview-react";
import "dockview-react/dist/styles/dockview.css";
import type { FunctionComponent } from "react";
import { CameraView } from "./views/CameraView";
import { CropsView } from "./views/CropsView";
import { CalibrationPanel, ControlsPanel, StatusPanel } from "./views/Panel";
import { PLOTS, Plot } from "./views/Plots";
import { Scene3D } from "./views/Scene3D";
import { ScreenView } from "./views/ScreenView";
import { VoicePanel } from "./views/VoicePanel";

// Every view is a dockview panel: drag tabs to rearrange, split or stack them, drag the dividers to resize.
// The layout is saved per browser; bump LAYOUT_KEY when panels change so stale layouts are dropped.
const LAYOUT_KEY = "gaze.layout.v4";

const components: Record<string, FunctionComponent<IDockviewPanelProps>> = {
  scene: () => <div className="pane flush"><Scene3D /></div>,
  camera: () => <CameraView />,
  crops: () => <CropsView />,
  screen: () => <ScreenView />,
  status: () => <StatusPanel />,
  calibration: () => <CalibrationPanel />,
  controls: () => <ControlsPanel />,
  voice: () => <VoicePanel />,
  ...Object.fromEntries(PLOTS.map((spec) => [`plot-${spec.id}`, () => <div className="pane flush"><Plot spec={spec} /></div>])),
};

function defaultLayout(api: DockviewApi) {
  api.addPanel({ id: "scene", component: "scene", title: "3D" });
  // columns first, then split them, so each column keeps its full height
  api.addPanel({ id: "camera", component: "camera", title: "Camera", position: { referencePanel: "scene", direction: "right" } });
  api.addPanel({ id: "screen", component: "screen", title: "Screen", position: { referencePanel: "camera", direction: "right" } });
  api.addPanel({ id: "status", component: "status", title: "Status", position: { referencePanel: "screen", direction: "right" } });
  api.addPanel({ id: "crops", component: "crops", title: "Model input", position: { referencePanel: "camera", direction: "below" } });
  api.addPanel({ id: "calibration", component: "calibration", title: "Calibration", position: { referencePanel: "status", direction: "below" } });
  api.addPanel({ id: "controls", component: "controls", title: "Controls", position: { referencePanel: "calibration", direction: "below" } });
  api.addPanel({ id: "voice", component: "voice", title: "Voice & agent", position: { referencePanel: "screen", direction: "below" } });
  let previous: string | undefined;
  for (const spec of PLOTS) {
    const id = `plot-${spec.id}`;
    api.addPanel({
      id, component: id, title: spec.title,
      position: previous ? { referencePanel: previous, direction: "right" } : { direction: "below" },
      ...(previous ? {} : { initialHeight: 210 }),
    });
    previous = id;
  }
  api.getPanel("scene")?.api.setSize({ width: Math.round(window.innerWidth * 0.38) });
  api.getPanel("status")?.api.setSize({ width: 280 });
}

let dockApi: DockviewApi | null = null;

export function resetLayout() {
  if (!dockApi) return;
  try {
    localStorage.removeItem(LAYOUT_KEY);
  } catch {
    // storage unavailable
  }
  dockApi.clear();
  defaultLayout(dockApi);
}

export function Layout() {
  const onReady = ({ api }: DockviewReadyEvent) => {
    dockApi = api;
    let restored = false;
    try {
      const saved = localStorage.getItem(LAYOUT_KEY);
      if (saved) {
        api.fromJSON(JSON.parse(saved));
        restored = true;
      }
    } catch {
      api.clear(); // corrupt or outdated layout: fall back to the default
    }
    if (!restored) defaultLayout(api);
    api.onDidLayoutChange(() => {
      try {
        localStorage.setItem(LAYOUT_KEY, JSON.stringify(api.toJSON()));
      } catch {
        // storage unavailable: the layout just won't be remembered
      }
    });
  };
  return <DockviewReact className="dock" components={components} onReady={onReady} theme={themeDark} />;
}
