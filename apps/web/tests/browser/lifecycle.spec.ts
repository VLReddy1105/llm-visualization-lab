import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

const mode = process.env.LLM_LAB_BROWSER_FAULT_MODE;
const outputDir = process.env.LLM_LAB_BROWSER_OUTPUT_DIR ?? "../../docs/validation";
test("real backend unavailable state", async ({ page }) => {
  test.skip(mode !== "offline", "Run with the actual backend stopped.");
  await page.goto("/");
  await expect(
    page.getByText("API unavailable", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Reconnect", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: /^Layer \d+$/ })).toHaveCount(
    0,
  );
  await page.screenshot({
    path: resolve(outputDir, "stage4-api-unavailable.png"),
    fullPage: true,
  });
});
test("real trace expiry without automatic retrace", async ({ page }) => {
  test.skip(
    mode !== "expiry",
    "Run against a real backend with a short configured TTL.",
  );
  let creations = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && /\/traces\?/.test(request.url()))
      creations += 1;
  });
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible({
    timeout: 60000,
  });
  await page.getByRole("button", { name: "TRACE MODEL", exact: true }).click();
  await expect(page.getByText("Trace ready", { exact: true })).toBeVisible();
  await expect(page.getByText("Trace expired", { exact: true })).toBeVisible({
    timeout: 15000,
  });
  await expect(
    page.getByRole("button", { name: "Run trace again", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: /^Layer \d+$/ })).toHaveCount(
    0,
  );
  expect(creations).toBe(1);
  await page.screenshot({
    path: resolve(outputDir, "stage4-expired.png"),
    fullPage: true,
  });
});
