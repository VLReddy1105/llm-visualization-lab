import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { TraceEnvelope, TraceStoreStatus } from "../../src/lib/api/client";

const apiUrl = "http://127.0.0.1:8000/api/v1";
const output = resolve("../../docs/validation");
async function screenshot(page: Page, name: string) {
  await page.screenshot({ path: resolve(output, name), fullPage: true });
}

test("real Qwen Model Journey, linked selection, navigation and presentation", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  const requests: string[] = [];
  page.on("request", (req) => {
    if (req.url().includes("/api/v1/traces"))
      requests.push(`${req.method()} ${req.url()}`);
  });
  const before: TraceStoreStatus = await (
    await request.get(`${apiUrl}/traces/status`)
  ).json();
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible({
    timeout: 60000,
  });
  await expect(page.getByLabel("Raw prompt")).toHaveValue(
    "The capital of Germany is",
  );
  await screenshot(page, "stage4-initial.png");
  const readyResponse = page.waitForResponse(async (response) => {
    if (
      response.request().method() !== "GET" ||
      !/\/traces\/[^/]+\?/.test(response.url())
    )
      return false;
    const envelope: TraceEnvelope = await response.json();
    return envelope.status === "ready";
  });
  await page.getByRole("button", { name: "TRACE MODEL" }).click();
  await expect(
    page.getByText("Tracing Qwen3-0.6B…", { exact: true }),
  ).toBeVisible();
  await screenshot(page, "stage4-loading.png");
  const response = await readyResponse;
  const trace: TraceEnvelope = await response.json();
  expect(trace.overview).not.toBeNull();
  const overview = trace.overview!;
  await expect(page.getByText("Trace ready", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Layer \d+$/ })).toHaveCount(
    28,
  );
  await expect(
    page.getByRole("region", { name: "Input tokens" }).getByRole("button"),
  ).toHaveCount(5);
  await expect(
    page.getByText("·Berlin", { exact: true }).first(),
  ).toBeVisible();
  expect(overview.prediction.top_k[0].decoded_token).toBe(" Berlin");
  expect(overview.prediction.top_k[0].probability).toBeCloseTo(0.477, 3);
  expect(overview.model.revision).toBe(
    "c1899de289a04d12100db370d81485cdf75e47ca",
  );
  expect(overview.model.architecture.layers).toBe(28);
  await screenshot(page, "stage4-model-journey.png");
  const rail = page.locator(".journey-viewport");
  await rail.hover();
  await page.mouse.wheel(0, 640);
  await expect
    .poll(() => rail.evaluate((node) => node.scrollLeft))
    .toBeGreaterThan(100);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await page.getByLabel("Jump to layer").selectOption("12");
  await expect(
    page.getByRole("heading", { name: "Layer 12", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Capture coverage", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Layer 12", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  const layerCenter = await page
    .locator('[data-node="layer-12"]')
    .boundingBox();
  const viewport = await page.locator(".journey-viewport").boundingBox();
  expect(
    Math.abs(
      layerCenter!.x +
        layerCenter!.width / 2 -
        (viewport!.x + viewport!.width / 2),
    ),
  ).toBeLessThan(10);
  await screenshot(page, "stage4-layer12.png");
  const token = page.getByRole("button", { name: 'Token 3: " Germany"' });
  await token.click();
  await expect(token).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("heading", { name: "Token 3", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("ĠGermany", { exact: true })).toBeVisible();
  await screenshot(page, "stage4-token-germany.png");
  await page
    .getByRole("button", { name: "Logits", exact: true })
    .first()
    .click();
  await expect(
    page.getByText(
      "Real next-token candidates · full-vocabulary probabilities",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.locator(".prediction-summary > span")).toHaveText("47.70%");
  await expect(token).toHaveAttribute("aria-pressed", "true");
  await screenshot(page, "stage4-logits.png");
  await page
    .getByRole("button", { name: "·Berlin", exact: true })
    .last()
    .click();
  await expect(page.getByText(/Selected candidate:/)).toBeVisible();
  await page
    .getByRole("button", { name: "Embedding", exact: true })
    .first()
    .click();
  await expect(page.getByText(/hidden dimensions 0–63/)).toBeVisible();
  await page.getByLabel("Jump to layer").selectOption("27");
  await expect(
    page.getByRole("heading", { name: "Layer 27", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Layer 27", exact: true }).focus();
  await page.keyboard.press("ArrowLeft");
  await expect(
    page.getByRole("button", { name: "Layer 26", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("Home");
  await expect(
    page.getByRole("heading", { name: "Input", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Fit Journey", exact: true }).click();
  await expect(page.locator(".model-journey")).toHaveClass(/compact/);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.locator(".model-journey")).toHaveClass(/comfortable/);
  // Both layers are already cached. Measure display opportunities without more inference.
  const performance = await page.evaluate(async () => {
    const durations: number[] = [];
    const nextFrame = () =>
      new Promise<void>((done) => requestAnimationFrame(() => done()));
    const selector = document.querySelector<HTMLSelectElement>(
      '[aria-label="Jump to layer"]',
    )!;
    for (let index = 0; index < 12; index++) {
      const layer = index % 2 ? 27 : 12;
      const start = window.performance.now();
      selector.value = String(layer);
      selector.dispatchEvent(new Event("change", { bubbles: true }));
      await nextFrame();
      await nextFrame();
      const selected = document.querySelector(`[data-node="layer-${layer}"]`);
      if (selected?.getAttribute("aria-pressed") !== "true")
        throw new Error("Cached selection failed");
      durations.push(window.performance.now() - start);
    }
    const sorted = [...durations].sort((a, b) => a - b);
    return {
      cached_layer_selection_two_frame_ms: durations,
      median_ms: sorted[Math.floor(sorted.length / 2)],
      max_ms: sorted.at(-1),
      journey_nodes: document.querySelectorAll(".journey-node").length,
      rendered_raw_values: document.querySelectorAll(".raw-data[open]").length,
      method:
        "12 cached layer changes; elapsed time through two requestAnimationFrame callbacks, reduced motion; exploratory browser observation, not a benchmark",
    };
  });
  const countBeforeEdit = requests.filter(
    (item) => item.startsWith("POST") && !item.includes("/slice"),
  ).length;
  await page.getByLabel("Raw prompt").fill("The capital of France is");
  await expect(page.getByText(/Prompt changed/)).toBeVisible();
  expect(
    requests.filter(
      (item) => item.startsWith("POST") && !item.includes("/slice"),
    ).length,
  ).toBe(countBeforeEdit);
  await page.getByLabel("Raw prompt").fill(overview.input.text);
  const axe = await new AxeBuilder({ page }).analyze();
  expect(axe.violations).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Model Journey", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await screenshot(page, "stage4-mobile.png");
  await page.setViewportSize({ width: 1440, height: 1100 });
  const after: TraceStoreStatus = await (
    await request.get(`${apiUrl}/traces/status`)
  ).json();
  expect(after.request_forward_passes - before.request_forward_passes).toBe(1);
  expect(errors).toEqual([]);
  const stableCount = requests.filter((item) =>
    /GET.*\/traces\/[^/]+\?/.test(item),
  ).length;
  await page.waitForTimeout(1100);
  expect(
    requests.filter((item) => /GET.*\/traces\/[^/]+\?/.test(item)).length,
  ).toBe(stableCount);
  writeFileSync(
    resolve(output, "stage4-browser-evidence.json"),
    JSON.stringify(
      {
        trace_id: trace.trace_id,
        tokens: overview.input.sequence_length,
        layers: overview.available.layers.length,
        model: overview.model,
        prediction: overview.prediction.top_k[0],
        requests,
        console_errors: errors,
        accessibility_violations: axe.violations,
        performance,
        retrieval_extra_forwards:
          after.request_forward_passes - before.request_forward_passes - 1,
        horizontal_center_error_px: Math.abs(
          layerCenter!.x +
            layerCenter!.width / 2 -
            viewport!.x -
            viewport!.width / 2,
        ),
      },
      null,
      2,
    ),
  );
  // A changed prompt is validated through the UI and a second actual Qwen computation.
  await page.getByLabel("Raw prompt").fill("The capital of France is");
  const fresh = page.waitForResponse(
    async (r) =>
      r.request().method() === "GET" &&
      /\/traces\/[^/]+\?/.test(r.url()) &&
      (await r.json()).status === "ready",
  );
  await page.getByRole("button", { name: "TRACE MODEL" }).click();
  const second: TraceEnvelope = await (await fresh).json();
  expect(second.trace_id).not.toBe(trace.trace_id);
  expect(second.overview!.input.text).toBe("The capital of France is");
  expect(second.overview!.prediction.top_k[0].token_id).not.toBe(
    overview.prediction.top_k[0].token_id,
  );
  await expect(
    page.getByRole("button", { name: 'Token 3: " France"' }),
  ).toHaveAttribute("aria-pressed", "false");
  writeFileSync(
    resolve(output, "stage4-second-prompt.json"),
    JSON.stringify(second, null, 2),
  );
  await expect(
    page.getByRole("region", { name: "Input tokens" }).getByRole("button"),
  ).toHaveCount(5);
  expect(errors).toEqual([]);
  await page.locator(".trace-details > summary").click();
  await page.getByRole("button", { name: "Delete trace", exact: true }).click();
  await expect(page.getByText("Trace deleted", { exact: true })).toBeVisible();
});
