import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    // Stylesheets are stubbed to nothing by default, which is fine for
    // component tests and useless for the one thing that has actually gone
    // wrong here: a rule that quietly out-ranked another. With this, a test
    // that imports `styles.css` gets it parsed into `document.styleSheets` and
    // can ask the real cascade what applies. Only `main.tsx` and
    // `Logs.layout.test.ts` import CSS, so nothing else changes.
    css: true,
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
