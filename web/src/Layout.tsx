import { DockviewReact, themeDark, type DockviewApi, type DockviewReadyEvent, type IDockviewPanelProps } from "dockview-react";
import "dockview-react/dist/styles/dockview.css";
import type { FunctionComponent } from "react";
import { CameraView } from "./views/CameraView";
import { CropsView } from "./views/CropsView";
import { CalibrationPanel, ControlsPanel, StatusPanel } from "./views/Panel";
import { PLOTS, Plot } from "./views/Plots";
import { Scene3D } from "./views/Scene3D";
import { ScreenView } from "./views/ScreenView";
import { HealthPanel } from "./views/HealthPanel";
import { LipsPlot } from "./views/LipsPlot";
import { VoicePanel } from "./views/VoicePanel";
import { ResearchPanel } from "./views/ResearchPanel";

// Every view is a dockview panel: drag tabs to rearrange, split or stack them, drag the dividers to resize.
// The layout is saved per browser; bump LAYOUT_KEY when panels change so stale layouts are dropped.
const LAYOUT_KEY = "pupil.layout.v10";

const components: Record<string, FunctionComponent<IDockviewPanelProps>> = {
  scene: () => <div className="pane flush"><Scene3D /></div>,
  camera: () => <CameraView />,
  crops: () => <CropsView />,
  screen: () => <ScreenView />,
  status: () => <StatusPanel />,
  calibration: () => <CalibrationPanel />,
  controls: () => <ControlsPanel />,
  voice: () => <VoicePanel />,
  research: () => <ResearchPanel />,
  lips: () => <LipsPlot />,
  health: () => <HealthPanel />,
  ...Object.fromEntries(PLOTS.map((spec) => [`plot-${spec.id}`, () => <div className="pane flush"><Plot spec={spec} /></div>])),
};

function defaultLayout(api: DockviewApi) {
  // Four columns by purpose: what the camera sees · where you look and what you said · eye health · the app.
  // Columns first, then split them, so each keeps its full height; "within" stacks panels as tabs.
  type Position = NonNullable<Parameters<DockviewApi["addPanel"]>[0]["position"]>;
  const add = (id: string, title: string, position?: Position, component = id) =>
    api.addPanel({ id, component, title, ...(position ? { position } : {}) });
  add("camera", "Camera");
  add("screen", "Screen", { referencePanel: "camera", direction: "right" });
  add("health", "Health", { referencePanel: "screen", direction: "right" });
  add("status", "Status", { referencePanel: "health", direction: "right" });
  add("crops", "Model input", { referencePanel: "camera", direction: "within" });
  add("scene", "3D", { referencePanel: "camera", direction: "below" });
  add("voice", "Voice & agent", { referencePanel: "screen", direction: "below" });
  add("lips", "Lips", { referencePanel: "voice", direction: "within" });
  add("research", "Decision research", { referencePanel: "voice", direction: "within" });
  add("plot-fatigue", "Fatigue (30 min)", { referencePanel: "health", direction: "below" });
  add("plot-eyes", "Eye openness", { referencePanel: "plot-fatigue", direction: "within" });
  add("calibration", "Calibration", { referencePanel: "status", direction: "within" });
  add("controls", "Controls", { referencePanel: "status", direction: "below" });
  // the bottom row: tracking plots, then model plots, each a group of tabs so they're wide enough to read
  add("plot-head", "Head (°)", { direction: "below" });
  add("plot-distance", "Distance (cm)", { referencePanel: "plot-head", direction: "within" });
  add("plot-lid", "Lid (°)", { referencePanel: "plot-head", direction: "within" });
  add("plot-disagreement", "ML vs eyeball (pt)", { referencePanel: "plot-head", direction: "right" });
  add("plot-engine", "Engine", { referencePanel: "plot-disagreement", direction: "within" });
  // first tab of each group in front
  for (const id of ["camera", "screen", "voice", "plot-fatigue", "status", "plot-head", "plot-disagreement"]) {
    api.getPanel(id)?.api.setActive();
  }
  api.getPanel("plot-head")?.api.setSize({ height: 170 });
  api.getPanel("camera")?.api.setSize({ width: Math.round(window.innerWidth * 0.3) });
  api.getPanel("status")?.api.setSize({ width: 260 });
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
