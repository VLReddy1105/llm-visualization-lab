import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  timeout: 90000,
  expect: { timeout: 15000 },
  fullyParallel: false,
  workers: 1,
  reporter: [
    ["list"],
    [
      "json",
      {
        outputFile: `../../docs/validation/${process.env.LLM_LAB_BROWSER_STAGE ?? "stage4"}-browser${process.env.LLM_LAB_BROWSER_FAULT_MODE ? "-" + process.env.LLM_LAB_BROWSER_FAULT_MODE : ""}-results.json`,
      },
    ],
  ],
  use: {
    baseURL: "http://127.0.0.1:3000",
    channel: "chrome",
    headless: true,
    viewport: { width: 1440, height: 1100 },
    reducedMotion: "reduce",
  },
});
