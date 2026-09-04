import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  build: {
    target: "esnext",
    rollupOptions: {
      // Relative to `root`; avoids __dirname, which an ESM config lacks.
      input: {
        settings: "index.html",
        overlay: "overlay.html",
      },
    },
  },
});
