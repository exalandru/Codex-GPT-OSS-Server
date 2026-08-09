import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// A fixed port because tauri.conf.json's devUrl has to match it, and 5274 keeps
// it clear of the diffusion server's 5273 when both are open.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5274, strictPort: true },
  build: { target: "safari15", outDir: "dist" },
});
