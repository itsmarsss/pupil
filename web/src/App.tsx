import { useEngineConnection } from "./engine";
import { Layout, resetLayout } from "./Layout";
import { Timeline } from "./views/Timeline";

export function App() {
  useEngineConnection();
  return (
    <div className="app">
      <header>
        <span className="title">Pupil dashboard</span>
        <Timeline />
        <button className="header-button" onClick={resetLayout}>Reset layout</button>
      </header>
      <main><Layout /></main>
    </div>
  );
}
