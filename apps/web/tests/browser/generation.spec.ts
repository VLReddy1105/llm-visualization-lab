import { test, expect, type APIRequestContext, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { GenerationSession } from "../../src/lib/api/client";

const api = "http://127.0.0.1:8000/api/v1";
const mode = process.env.LLM_LAB_BROWSER_FAULT_MODE;
const output = resolve(process.env.LLM_LAB_BROWSER_OUTPUT_DIR ?? "../../docs/validation");
const save = (name: string, value: unknown) =>
  writeFileSync(resolve(output, name), `${JSON.stringify(value, null, 2)}\n`);
const forwardCount = async (request: APIRequestContext): Promise<number> =>
  (await (await request.get(`${api}/traces/status`)).json()).request_forward_passes;

async function waitForTerminal(page: Page, status: "completed" | "stopped" | "cancelled" | "failed") {
  await expect(page.locator(".generation-status")).toContainText(status, { timeout: 180000 });
}

async function ready(page: Page) {
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible({ timeout: 90000 });
  await page.getByRole("button", { name: "Generation Journey", exact: true }).click();
}

test("real Germany greedy generation links exact steps to original inspectors", async ({ page, request }) => {
  test.skip(!!mode, "Normal real-backend acceptance only.");
  test.setTimeout(240000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await ready(page);
  const before = await forwardCount(request);
  await page.getByLabel("Raw prompt").fill("The capital of Germany is");
  await page.getByLabel("Max new tokens").fill("2");
  const creation = page.waitForResponse((response) =>
    response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/generations",
  );
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  const created = await (await creation).json() as GenerationSession;
  expect(created.original_prompt).toBe("The capital of Germany is");
  expect(created.original_input_ids.length).toBeGreaterThan(0);
  expect(created.execution_configuration).toMatchObject({ input_mode: "raw", add_special_tokens: false, use_cache: false });
  await waitForTerminal(page, "completed");
  const sessionId = created.generation_id;
  const sessionResponse = await request.get(`${api}/generations/${sessionId}`);
  expect(sessionResponse.ok()).toBe(true);
  const session: GenerationSession = await sessionResponse.json();
  const steps = session.steps ?? [];
  const selectedTokenIds = session.selected_token_ids ?? [];
  expect(session.status).toBe("completed");
  expect(session.stop_reason).toBe("max_new_tokens");
  expect(session.attempted_forwards).toBe(2);
  expect(session.completed_steps).toBe(session.attempted_forwards);
  expect(selectedTokenIds).toEqual(steps.map((step) => step.selected_token_id));
  expect(session.decoded_continuation).toBe(steps.at(-1)?.decoded_continuation);
  expect(await forwardCount(request)).toBe(before + 2);

  for (const step of steps) {
    expect(step.prefix_token_ids).toEqual([
      ...session.original_input_ids,
      ...selectedTokenIds.slice(0, step.step_index),
    ]);
    expect(step.decision_position).toBe(step.prefix_token_ids.length - 1);
    expect(step.selected_position).toBe(step.prefix_token_ids.length);
    expect(step.forward_completion_status).toBe("completed");
    expect(step.forward_execution_id).toBeTruthy();
    expect(step.source_trace_id).toBeTruthy();
    const original = await request.get(`${api}/traces/${step.source_trace_id}`);
    expect(original.ok()).toBe(true);
    const trace = await original.json();
    expect(trace.overview.runtime).toMatchObject({
      generation_id: sessionId,
      generation_step: step.step_index,
      forward_execution_id: step.forward_execution_id,
      exact_input_ids: true,
      forward_count: 1,
    });
    expect(trace.overview.capture).toMatchObject({ profile: "deep", layers: [12], token_indices: [step.decision_position] });
  }

  const last = steps.at(-1)!;
  const lastOption = page.getByRole("option", { name: new RegExp(`ID ${last.selected_token_id}`) });
  await expect(lastOption).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("heading", { name: `Selected token ID ${last.selected_token_id}` })).toBeVisible();
  await expect(page.getByText(`ID ${last.raw_argmax_token_id}${last.raw_argmax_token_id === last.selected_token_id ? " (selected)" : " (different from selected)"}`)).toBeVisible();
  const probability = page.locator(".generation-candidate").filter({ hasText: `ID ${last.raw_argmax_token_id}` }).first();
  await expect(probability).toContainText(last.selected_model_probability.toPrecision(5));

  const beforeInspection = await forwardCount(request);
  await page.getByRole("button", { name: "Residual Stream", exact: true }).click();
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", last.source_trace_id!);
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-token-index", String(last.decision_position));
  await page.getByRole("navigation", { name: "Residual and MLP views" }).getByRole("button", { name: "MLP Explorer", exact: true }).click();
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-kind", "mlp_internals");
  await expect(page.getByTestId("block-analysis")).toHaveAttribute("data-trace-id", last.source_trace_id!);
  expect(await forwardCount(request)).toBe(beforeInspection);

  await page.getByRole("button", { name: "Generation Journey", exact: true }).click();
  const timeline = page.getByRole("listbox", { name: "Generation steps" });
  await page.getByRole("option", { name: new RegExp(`ID ${last.selected_token_id}`) }).focus();
  await page.keyboard.press("Home");
  await expect(page.getByRole("heading", { name: `Selected token ID ${steps[0].selected_token_id}` })).toBeVisible();
  await page.keyboard.press("End");
  await expect(page.getByRole("heading", { name: `Selected token ID ${last.selected_token_id}` })).toBeVisible();
  await expect(timeline).toBeVisible();

  const axe = await new AxeBuilder({ page }).analyze();
  expect(axe.violations).toEqual([]);
  await page.screenshot({ path: resolve(output, "stage8-germany-greedy-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: resolve(output, "stage8-germany-greedy-mobile.png"), fullPage: true });
  expect(await forwardCount(request)).toBe(before + 2);
  expect(pageErrors).toEqual([]);
  save("stage8-germany-greedy-browser.json", {
    generation_id: sessionId,
    source_trace_ids: steps.map((step) => step.source_trace_id),
    selected_ids: selectedTokenIds,
    decoded_continuation: session.decoded_continuation,
    forward_counter_before: before,
    forward_counter_after_exploration: await forwardCount(request),
    exploration_forwards: 0,
    page_errors: pageErrors,
    accessibility: { violations: axe.violations, passes: axe.passes.length },
    session_payload_bytes: session.result_bytes,
  });
});

test("real France sampled generation reports selected token separately from argmax", async ({ page, request }) => {
  test.skip(!!mode, "Normal real-backend acceptance only.");
  test.setTimeout(240000);
  const evidence = JSON.parse(readFileSync(resolve(process.cwd(), "../../docs/validation/stage8-scientific-cpu.json"), "utf8"));
  const divergence = evidence.greedy_and_sampling_runs.find((run: { mode: string; differs_from_argmax_seed_case?: boolean }) =>
    run.mode === "sample" && run.differs_from_argmax_seed_case,
  );
  expect(divergence).toBeTruthy();
  const seed = divergence.seed as number;
  await ready(page);
  const before = await forwardCount(request);
  await page.getByLabel("Raw prompt").fill("The capital of France is");
  await page.getByLabel("Max new tokens").fill("2");
  await page.getByLabel("Generation mode").selectOption("sample");
  await page.getByLabel("Temperature").fill("0.7");
  await page.getByLabel("Top-k").fill("40");
  await page.getByLabel("Top-p").fill("0.9");
  await page.getByLabel("Sampling seed").fill(String(seed));
  const creation = page.waitForResponse((response) =>
    response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/generations",
  );
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  const created = await (await creation).json() as GenerationSession;
  await waitForTerminal(page, "completed");
  const response = await request.get(`${api}/generations/${created.generation_id}`);
  const session: GenerationSession = await response.json();
  expect(session.sampling).toMatchObject({ mode: "sample", temperature: 0.7, top_k: 40, top_p: 0.9, seed });
  expect(session.completed_steps).toBe(2);
  const steps = session.steps ?? [];
  const selectedTokenIds = session.selected_token_ids ?? [];
  expect(steps.some((step) => step.selected_token_id !== step.raw_argmax_token_id)).toBe(true);
  expect(await forwardCount(request)).toBe(before + 2);
  const first = steps[0];
  await page.getByRole("listbox", { name: "Generation steps" })
    .getByRole("option", { name: new RegExp(`ID ${first.selected_token_id}`) }).click();
  await expect(page.getByRole("heading", { name: `Selected token ID ${first.selected_token_id}` })).toBeVisible();
  await expect(page.getByText("Adjusted probability")).toBeVisible();
  await expect(page.getByRole("button", { name: "Attention Microscope", exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(output, "stage8-france-sampled-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: resolve(output, "stage8-france-sampled-mobile.png"), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByRole("button", { name: "Attention Microscope", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Attention matrix" })).toBeVisible();
  expect(await forwardCount(request)).toBe(before + 2);
  save("stage8-france-sampled-browser.json", {
    generation_id: session.generation_id,
    source_trace_ids: steps.map((step) => step.source_trace_id),
    selected_ids: selectedTokenIds,
    argmax_ids: steps.map((step) => step.raw_argmax_token_id),
    selected_differs_from_argmax: steps.some((step) => step.selected_token_id !== step.raw_argmax_token_id),
    forward_counter_before: before,
    forward_counter_after_exploration: await forwardCount(request),
    exploration_forwards: 0,
    seed,
    sample_parameters: session.sampling,
    session_payload_bytes: session.result_bytes,
  });
});

test("cancellation stops before publishing another generation step", async ({ page, request }) => {
  test.skip(!!mode, "Normal real-backend acceptance only.");
  test.setTimeout(180000);
  await ready(page);
  const before = await forwardCount(request);
  await page.getByLabel("Raw prompt").fill("In a laboratory, a scientist carefully records each observation before changing one variable.");
  await page.getByLabel("Max new tokens").fill("16");
  await page.getByLabel("Capture up to four original step traces").uncheck();
  const creation = page.waitForResponse((response) =>
    response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/generations",
  );
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  const created = await (await creation).json() as GenerationSession;
  const cancel = page.getByRole("button", { name: "Cancel after current forward" });
  await cancel.waitFor({ state: "visible", timeout: 60000 });
  await cancel.click();
  await waitForTerminal(page, "cancelled");
  const result = await (await request.get(`${api}/generations/${created.generation_id}`)).json() as GenerationSession;
  expect(result.status).toBe("cancelled");
  expect(result.cancellation_requested).toBe(true);
  expect(result.completed_steps).toBeLessThan(result.max_new_tokens);
  const after = await forwardCount(request);
  await page.waitForTimeout(800);
  expect(await forwardCount(request)).toBe(after);
  save("stage8-cancellation-browser.json", {
    generation_id: result.generation_id,
    status: result.status,
    stop_reason: result.stop_reason,
    completed_steps: result.completed_steps,
    attempted_forwards: result.attempted_forwards,
    counter_before: before,
    counter_after: after,
    no_forward_after_cancellation: true,
  });
});

test("invalid generation settings are rejected in the browser before inference", async ({ page, request }) => {
  test.skip(!!mode, "Normal real-backend acceptance only.");
  await ready(page);
  const before = await forwardCount(request);
  await page.getByLabel("Generation mode").selectOption("sample");
  await page.getByLabel("Sampling seed").fill("-1");
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  await expect(page.locator(".generation-error[role='alert']")).toContainText("Seed must be a nonnegative");
  expect(await forwardCount(request)).toBe(before);
});

test("generation session expires without retaining stale decisions", async ({ page }) => {
  test.skip(mode !== "generation-expiry", "Run with an actual short generation-session TTL.");
  test.setTimeout(90000);
  await ready(page);
  await page.getByLabel("Raw prompt").fill("The capital of Germany is");
  await page.getByLabel("Max new tokens").fill("1");
  await page.getByLabel("Capture up to four original step traces").uncheck();
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  await waitForTerminal(page, "completed");
  await expect(page.getByText(/This generation session expired/)).toBeVisible({ timeout: 30000 });
  await expect(page.getByRole("heading", { name: "Selected token ID" })).toHaveCount(0);
  await page.screenshot({ path: resolve(output, "stage8-generation-expired.png"), fullPage: true });
});

test("expired original step traces become unavailable without recapture", async ({ page, request }) => {
  test.skip(mode !== "generation-source-expiry", "Run with an actual short trace-store TTL.");
  test.setTimeout(90000);
  await ready(page);
  await page.getByLabel("Raw prompt").fill("The capital of Germany is");
  await page.getByLabel("Max new tokens").fill("1");
  const creation = page.waitForResponse((response) =>
    response.request().method() === "POST" && new URL(response.url()).pathname === "/api/v1/generations",
  );
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  const created = await (await creation).json() as GenerationSession;
  await waitForTerminal(page, "completed");
  const session = await (await request.get(`${api}/generations/${created.generation_id}`)).json() as GenerationSession;
  const step = session.steps![0];
  const forwardsAfterGeneration = await forwardCount(request);
  expect(step.source_trace_id).toBeTruthy();
  await expect(page.getByRole("button", { name: "Attention Microscope", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Attention Microscope", exact: true })).toHaveCount(0, { timeout: 30000 });
  await expect(page.getByText(/original step trace expired/)).toBeVisible();
  await expect(page.getByRole("heading", { name: `Selected token ID ${step.selected_token_id}` })).toBeVisible();
  expect(await forwardCount(request)).toBe(forwardsAfterGeneration);
  await page.screenshot({ path: resolve(output, "stage8-source-trace-expired.png"), fullPage: true });
  save("stage8-source-trace-expired-browser.json", {
    generation_id: session.generation_id,
    source_trace_id: step.source_trace_id,
    retained_step_id: step.step_id,
    selected_token_id: step.selected_token_id,
    capture_unavailable_reason: step.capture_unavailable_reason,
    inspector_removed_after_expiry: true,
    recaptured: false,
    forwards_after_expiry: await forwardCount(request),
  });
});

test("generation handles the unavailable backend", async ({ page }) => {
  test.skip(mode !== "offline", "Run with the actual backend stopped.");
  await page.goto("/");
  await page.getByRole("button", { name: "Generation Journey", exact: true }).click();
  await page.getByRole("button", { name: /START GENERATION/i }).click();
  await expect(page.locator(".generation-error[role='alert']")).toContainText("Cannot reach FastAPI");
  await page.screenshot({ path: resolve(output, "stage8-generation-offline.png"), fullPage: true });
});
