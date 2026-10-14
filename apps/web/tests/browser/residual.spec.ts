import { test, expect, type Page, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { AnalysisJob, TraceEnvelope } from "../../src/lib/api/client";

const api = "http://127.0.0.1:8000/api/v1";
test.use({ actionTimeout: 15000 });
const output = resolve(process.env.LLM_LAB_BROWSER_OUTPUT_DIR ?? "../../docs/validation");
const save = (name: string, value: unknown) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n");
const counter = async (request: APIRequestContext): Promise<number> =>
  (await (await request.get(`${api}/traces/status`)).json()).request_forward_passes;

async function overview(page: Page, country: string): Promise<TraceEnvelope> {
  await page.getByLabel("Raw prompt").fill(`The capital of ${country} is`);
  const response = page.waitForResponse(async (r) => {
    if (r.request().method() !== "GET" || !/\/traces\/[^/]+\?/.test(r.url())) return false;
    return (await r.json()).status === "ready";
  });
  await page.getByRole("button", { name: "TRACE MODEL", exact: true }).click();
  const trace = await (await response).json();
  await expect(page.getByText("Trace ready", { exact: true })).toBeVisible();
  return trace;
}

test("Stage 7 real analysis expiry, saved-data recompute and source expiry", async ({ page, request }) => {
  test.skip(process.env.LLM_LAB_BROWSER_FAULT_MODE !== "analysis-expiry", "Run with an actual short analysis/source TTL backend.");
  test.setTimeout(90000);
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible({ timeout: 60000 });
  const before = await counter(request);
  const source = await overview(page, "Germany");
  await page.getByRole("button", { name: 'Token 4: " is"', exact: true }).click();
  await page.getByLabel("Jump to layer").selectOption("12");
  await page.getByRole("navigation", { name: "Residual and MLP views" }).getByRole("button", { name: "Residual Stream", exact: true }).click();
  const detailed = await capture(page, 12, 4);
  expect(await counter(request)).toBe(before + 2);
  await expect(page.locator(".error-notice")).toContainText("ANALYSIS_EXPIRED", { timeout: 15000 });
  await expect(page.getByTestId("block-analysis")).toHaveCount(0);
  await page.locator(".inspector-panel").screenshot({ path: resolve(output, "stage7-analysis-expired.png") });
  await page.getByRole("button", { name: "Recompute saved-data analysis", exact: true }).click();
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", detailed);
  expect(await counter(request)).toBe(before + 2);
  await expect(page.getByText("Trace expired", { exact: true })).toBeVisible({ timeout: 45000 });
  await expect(page.getByTestId("block-analysis")).toHaveCount(0);
  expect(await counter(request)).toBe(before + 2);
  await page.screenshot({ path: resolve(output, "stage7-source-expired.png"), fullPage: true });
  save("stage7-expiry-evidence.json", { source: source.trace_id, detailed, before, after: await counter(request), explicitCaptures: 2, explorationForwards: 0 });
});

async function capture(page: Page, layer: number, token: number): Promise<string> {
  const response = page.waitForResponse(
    (r) => r.request().method() === "POST" && /\/traces\?/.test(r.url()),
  );
  await page.getByRole("button", { name: new RegExp(`^Capture Layer ${layer} .*`) }).click();
  const r = await response;
  expect(r.request().postDataJSON().capture).toEqual({ profile: "mlp", layers: [layer], tokens: [token] });
  const trace: TraceEnvelope = await r.json();
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", trace.trace_id);
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-layer-index", String(layer));
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-token-index", String(token));
  return trace.trace_id;
}

for (const country of ["Germany", "France"]) {
  test(`Stage 7 real ${country}: residuals, gated MLP, capture scope and zero exploration forwards`, async ({ page, request }) => {
    test.skip(!!process.env.LLM_LAB_BROWSER_FAULT_MODE, "Normal real-backend validation.");
    test.setTimeout(180000);
    const errors: string[] = [];
    const jobs: AnalysisJob[] = [];
    const fixtures: { method: string; path: string; body: unknown; response: unknown }[] = [];
    const reads: Promise<void>[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("response", (r) => {
      if (!r.url().startsWith(api) || r.status() >= 400) return;
      reads.push((async () => {
        const body = await r.json();
        fixtures.push({ method: r.request().method(), path: new URL(r.url()).pathname + new URL(r.url()).search,
          body: r.request().postDataJSON(), response: body });
        if (body.analysis_id && body.status === "ready") jobs.push(body);
      })());
    });
    await page.goto("/");
    await expect(page.getByText("Model ready", { exact: true })).toBeVisible({ timeout: 60000 });
    const before = await counter(request);
    const source = await overview(page, country);
    await page.getByRole("button", { name: 'Token 4: " is"', exact: true }).click();
    await page.getByLabel("Jump to layer").selectOption("12");
    await page.getByRole("navigation", { name: "Residual and MLP views" }).getByRole("button", { name: "Residual Stream", exact: true }).click();
    await expect(page.getByTestId("block-analysis")).toHaveCount(0);
    expect(await counter(request)).toBe(before + 1);
    const detailed = await capture(page, 12, 4);
    expect(detailed).not.toBe(source.trace_id);

    // Remaining interaction checks use the production UI and real saved-data jobs.
    // No response interception or scientific fixtures are installed in this test.
    await Promise.all(reads);
    const first = jobs.find((j) => j.trace_id === detailed && j.result?.kind === "residual_stream");
    expect(first?.result?.source.trace_id).toBe(detailed);
    expect(first?.result?.source.token.token_index).toBe(4);
    if (first?.result?.kind !== "residual_stream") throw new Error("Missing real residual analysis");
    expect(first.result.operations).toHaveLength(7);
    expect(first.result.reconciliation.every((item) => item.metrics.passed)).toBe(true);
    expect(first.result.operations.every((item) => item.provenance === "raw")).toBe(true);
    for (const symbol of ["r0", "r1", "n_mlp", "r2"]) {
      await page.locator(`[data-operation="${symbol}"]`).click();
      await expect(page.locator(`[data-operation="${symbol}"]`)).toHaveAttribute("aria-pressed", "true");
      await expect(page.getByTestId("block-operation-detail")).toContainText(`SELECTED OPERATION · ${symbol}`);
    }
    await page.locator('[data-operation="r1"]').focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.locator('[data-operation="n_mlp"]')).toBeFocused();
    await page.keyboard.press("End");
    await expect(page.locator('[data-operation="output"]')).toHaveAttribute("aria-pressed", "true");
    await page.locator('[data-operation="r1"]').click();
    await page.locator(".inspector-panel").screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-residual.png`) });
    const residualAxe = await new AxeBuilder({ page }).analyze();
    expect(residualAxe.violations).toEqual([]);

    await page.getByRole("navigation", { name: "Residual and MLP views" }).getByRole("button", { name: "MLP Explorer", exact: true }).click();
    await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-kind", "mlp_internals");
    for (const symbol of ["g", "u", "s", "p", "m", "r2"]) {
      await page.locator(`[data-operation="${symbol}"]`).click();
      await expect(page.getByTestId("block-operation-detail")).toContainText(`SELECTED OPERATION · ${symbol}`);
    }
    await page.locator('[data-operation="p"]').click();
    await page.getByLabel("Hidden coordinate", { exact: true }).fill("19");
    await page.getByLabel("Intermediate coordinate", { exact: true }).fill("17");
    await page.getByLabel("Coordinate count", { exact: true }).fill("7");
    await page.getByLabel("Ranking", { exact: true }).selectOption("negative");
    await page.getByRole("button", { name: "Apply coordinates", exact: true }).click();
    await expect(page.getByText("EXACT INTERMEDIATE COORDINATE 17", { exact: true })).toBeVisible();
    await Promise.all(reads);
    const selected = jobs.findLast((j) => j.trace_id === detailed && j.result?.kind === "mlp_internals" && j.result.selected_gate_coordinate.index === 17);
    if (selected?.result?.kind !== "mlp_internals") throw new Error("Missing selected real MLP result");
    const gate = selected.result.selected_gate_coordinate;
    const slices: Record<string, unknown> = {};
    for (const [field, component] of [["gate", "gate"], ["activated_gate", "activated_gate"], ["up", "up"], ["product", "gated"]] as const) {
      const response = await request.post(`${api}/traces/${detailed}/components/${encodeURIComponent(`layer[12].mlp.${component}`)}/slice`, {
        data: { selectors: { batch: { index: 0 }, token: { index: 4 }, intermediate: { index: 17 } }, format: "json" },
      });
      expect(response.ok()).toBe(true);
      const slice = await response.json();
      expect(slice.trace_id).toBe(detailed);
      expect(slice.values).toEqual([gate[field]]);
      slices[field] = slice;
    }
    expect(selected.result.gate_coordinates).toHaveLength(7);
    expect(selected.result.gate_coordinates.every((c) => c.product < 0)).toBe(true);
    await expect(page.locator(".gate-coordinates tbody tr")).toHaveCount(7);
    const formatted = (v: number) => v !== 0 && Math.abs(v) < 0.0001 ? v.toExponential(4) : v.toLocaleString("en", { maximumSignificantDigits: 7 });
    for (const [label, value] of [["Raw gate", gate.gate], ["SiLU(gate)", gate.activated_gate], ["Up projection", gate.up], ["Product", gate.product]] as const)
      await expect(page.locator(".gate-coordinates dl div").filter({ has: page.locator("dt", { hasText: new RegExp(`^${label.replace(/[()]/g, "\\$&")}$`) }) }).locator("dd")).toHaveText(formatted(value));
    await page.locator(".inspector-panel").screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-mlp.png`) });
    const mlpAxe = await new AxeBuilder({ page }).analyze();
    expect(mlpAxe.violations).toEqual([]);
    // Invalid coordinate is rejected locally and never causes a forward or synthetic value.
    await page.getByLabel("Intermediate coordinate", { exact: true }).fill("3072");
    await page.getByRole("button", { name: "Apply coordinates", exact: true }).click();
    await expect(page.locator(".block-input-error")).toContainText("Use hidden coordinates");
    await page.getByLabel("Intermediate coordinate", { exact: true }).fill("17");
    await page.getByRole("button", { name: "Apply coordinates", exact: true }).click();
    await page.setViewportSize({ width: 800, height: 1100 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    await page.locator(".inspector-panel").screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-narrow.png`) });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    await page.screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-mobile.png`), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1100 });
    expect(await counter(request)).toBe(before + 2);
    await page.getByRole("navigation", { name: "Inspector breadcrumb" }).getByRole("button", { name: "Model Journey", exact: true }).click();
    await page.getByRole("navigation", { name: "Residual and MLP views" }).getByRole("button", { name: "Residual Stream", exact: true }).click();
    await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", detailed);
    await page.getByRole("button", { name: 'Token 0: "The"', exact: true }).click();
    await expect(page.getByTestId("block-analysis")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Capture Layer 12 Token 0 Details", exact: true })).toBeVisible();
    expect(await counter(request)).toBe(before + 2);
    const tokenZero = await capture(page, 12, 0);
    await page.getByLabel("Jump to layer").selectOption("27");
    await expect(page.getByTestId("block-analysis")).toHaveCount(0);
    expect(await counter(request)).toBe(before + 3);
    const layer27 = await capture(page, 27, 0);
    await page.locator(".inspector-panel").screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-layer27-token0.png`) });
    await page.getByLabel("Jump to layer").selectOption("12");
    await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", tokenZero);
    await page.getByRole("button", { name: 'Token 4: " is"', exact: true }).click();
    await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", detailed);
    expect(await counter(request)).toBe(before + 4);
    // Actual deletion, followed by an uncached saved-data request, removes stale science.
    expect((await request.delete(`${api}/traces/${detailed}`)).status()).toBe(204);
    await page.getByLabel("Hidden coordinate", { exact: true }).fill("23");
    await page.getByRole("button", { name: "Apply coordinates", exact: true }).click();
    await expect(page.locator(".error-notice")).toContainText(/TRACE_DELETED|TRACE_NOT_FOUND/);
    await expect(page.getByTestId("block-analysis")).toHaveCount(0);
    await page.locator(".inspector-panel").screenshot({ path: resolve(output, `stage7-${country.toLowerCase()}-deleted.png`) });
    expect(await counter(request)).toBe(before + 4);
    const recovery = await capture(page, 12, 4);
    expect(recovery).not.toBe(detailed);
    expect(await counter(request)).toBe(before + 5);
    expect(errors).toEqual([]);
    await Promise.all(reads);
    save(`stage7-${country.toLowerCase()}-browser.json`, { source, detailed, tokenZero, layer27, recovery, slices, before, after: await counter(request), explicitCaptures: 5, explorationForwards: 0, jobs,
      accessibility: { residual: { violations: residualAxe.violations, passes: residualAxe.passes.length }, mlp: { violations: mlpAxe.violations, passes: mlpAxe.passes.length } }, errors });
    save(`stage7-${country.toLowerCase()}-http-fixtures.json`, fixtures);
  });
}
