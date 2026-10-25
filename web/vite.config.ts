import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `pnpm dev` serves the dashboard with hot reload and forwards the engine's socket from a running
// `gaze --dashboard` (port 8766). `pnpm build` writes web/dist, which the Python server serves directly.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/ws": { target: "ws://127.0.0.1:8766", ws: true } } },
  build: { chunkSizeWarningLimit: 1600 }, // three.js; a local tool, so one bundle is fine
});
