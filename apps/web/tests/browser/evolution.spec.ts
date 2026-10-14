import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { AnalysisJob, TraceEnvelope } from "../../src/lib/api/client";

const api = "http://127.0.0.1:8000/api/v1";
const output = resolve(process.env.LLM_LAB_BROWSER_OUTPUT_DIR ?? "../../docs/validation");
const save = (name: string, value: unknown) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n");
async function run(page: Page, prompt: string) {
  await page.getByLabel("Raw prompt").fill(prompt);
  const ready = page.waitForResponse(
    async (r) =>
      r.request().method() === "GET" &&
      /\/traces\/[^/]+\?/.test(r.url()) &&
      (await r.json()).status === "ready",
    { timeout: 60000 },
  );
  await page.getByRole("button", { name: "TRACE MODEL" }).click();
  const trace: TraceEnvelope = await (await ready).json();
  await expect(page.getByText("Trace ready", { exact: true })).toBeVisible();
  return trace;
}
async function view(page: Page, name: string) {
  await page
    .getByRole("navigation", { name: "Token evolution views" })
    .getByRole("button", { name, exact: true })
    .click();
}

for (const country of ["Germany", "France"])
  test(`Stage 6 real ${country}: depth, lens, sampling, reuse and accessibility`, async ({
    page,
    request,
  }) => {
    test.setTimeout(180000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const fixtures: {
      method: string;
      path: string;
      body: unknown;
      response: unknown;
    }[] = [];
    const jobs: AnalysisJob[] = [];
    const reads: Promise<void>[] = [];
    page.on("response", (r) => {
      if (!r.url().startsWith(api) || r.status() >= 400) return;
      reads.push(
        (async () => {
          const body = await r.json();
          fixtures.push({
            method: r.request().method(),
            path: new URL(r.url()).pathname + new URL(r.url()).search,
            body: r.request().postDataJSON(),
            response: body,
          });
          if (body.analysis_id && body.status === "ready") jobs.push(body);
        })(),
      );
    });
    await page.goto("/");
    await expect(page.getByText("Model ready", { exact: true })).toBeVisible({
      timeout: 60000,
    });
    const before = (await (await request.get(api + "/traces/status")).json())
      .request_forward_passes;
    const trace = await run(page, `The capital of ${country} is`);
    await view(page, "Token Journey");
    const timeline = page.getByRole("toolbar", {
      name: "Transformer depth checkpoints",
    });
    await expect(timeline.getByRole("button")).toHaveCount(
      trace.overview!.model.architecture.layers + 2,
    );
    await expect(page.getByTestId("depth-analysis")).toHaveAttribute(
      "data-token-index",
      String(trace.overview!.input.sequence_length - 1),
    );
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(output, `stage6-${country.toLowerCase()}-journey.png`),
      });
    await timeline
      .getByRole("button", { name: "POST-BLOCK Layer 12", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Open Layer 12 in Microscope" })
      .click();
    await expect(
      page.getByText("Two residual additions. One continuous representation."),
    ).toBeVisible();
    await view(page, "Representation Evolution");
    await expect(
      timeline.getByRole("button", {
        name: "POST-BLOCK Layer 12",
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("depth-analysis")).toHaveAttribute(
      "data-token-index",
      "4",
    );
    await page.getByLabel("Compare active checkpoint with").selectOption("1");
    await expect(
      page.getByText("Layer 0 → Layer 12 · same token, same trace"),
    ).toBeVisible();
    const current = timeline.getByRole("button", {
      name: "POST-BLOCK Layer 12",
      exact: true,
    });
    await current.focus();
    await page.keyboard.press("ArrowRight");
    await expect(
      timeline.getByRole("button", {
        name: "POST-BLOCK Layer 13",
        exact: true,
      }),
    ).toBeFocused();
    await page.keyboard.press("End");
    await expect(
      timeline.getByRole("button", { name: "READOUT Final norm", exact: true }),
    ).toHaveAttribute("aria-pressed", "true");
    await page.locator(".depth-chart circle[data-depth='13']").first().click();
    await expect(
      timeline.getByRole("button", {
        name: "POST-BLOCK Layer 12",
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", "true");
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(
          output,
          `stage6-${country.toLowerCase()}-representation.png`,
        ),
      });
    await view(page, "Logit Lens");
    await expect(
      page.getByText("Final projection agrees with retained logits", {
        exact: true,
      }),
    ).toBeVisible({ timeout: 60000 });
    await Promise.all(reads);
    const lens = jobs.find(
      (j) => j.trace_id === trace.trace_id && j.result?.kind === "logit_lens",
    )!;
    expect(lens?.result?.kind).toBe("logit_lens");
    if (lens.result?.kind !== "logit_lens")
      throw new Error("Missing actual lens");
    expect(lens.result.final_logits_comparison?.passed).toBe(true);
    expect(lens.result.final_normalization_comparison.passed).toBe(true);
    const top = lens.result.checkpoints[13].top_candidates[0];
    await expect(
      page
        .getByRole("region", { name: "Selected checkpoint projections" })
        .getByText(String(top.token_id), { exact: true }),
    ).toBeVisible();
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(output, `stage6-${country.toLowerCase()}-lens.png`),
      });
    await page
      .getByRole("button", { name: `Pin #${top.token_id}`, exact: true })
      .click();
    await expect(
      page.getByText("Final projection agrees with retained logits", {
        exact: true,
      }),
    ).toBeVisible();
    await view(page, "Candidate Evolution");
    await expect(
      page.getByRole("region", {
        name: "Candidate Probability Evolution",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByLabel("Candidate for full-vocabulary rank")
      .selectOption(String(top.token_id));
    await page
      .getByText("Layer Prediction Table · diagnostic projections", {
        exact: true,
      })
      .click();
    await expect(page.locator(".layer-predictions tbody tr")).toHaveCount(
      trace.overview!.model.architecture.layers + 2,
    );
    await page
      .getByText("Layer Prediction Table · diagnostic projections", {
        exact: true,
      })
      .click();
    const accessibility = await new AxeBuilder({ page }).analyze();
    expect(accessibility.violations).toEqual([]);
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(output, `stage6-${country.toLowerCase()}-candidates.png`),
      });
    // Earlier input position has its own next-position interpretation and no false output comparison.
    await page
      .getByRole("button", { name: `Token 0: "The"`, exact: true })
      .click();
    await expect(page.getByTestId("depth-analysis")).toHaveAttribute(
      "data-token-index",
      "0",
      { timeout: 60000 },
    );
    await expect(
      page.getByText(
        "Original logits comparison unavailable at this position.",
        { exact: true },
      ),
    ).toBeVisible();
    await page
      .getByRole("button", { name: 'Token 4: " is"', exact: true })
      .click();
    await expect(
      page.getByText("Final projection agrees with retained logits", {
        exact: true,
      }),
    ).toBeVisible();
    await view(page, "Sampling Lab");
    await expect(page.locator(".sampling-applied")).toContainText(
      "temperature 1",
    );
    await page.getByLabel("Temperature", { exact: true }).fill("0.7");
    await page.getByLabel("Top-k (0 disables)", { exact: true }).fill("20");
    await page.getByLabel("Top-p", { exact: true }).fill("0.8");
    await page
      .getByRole("button", { name: "Apply preview", exact: true })
      .click();
    await expect(page.locator(".sampling-applied")).toContainText(
      "temperature 0.7 · top-k 20 · top-p 0.8",
    );
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(output, `stage6-${country.toLowerCase()}-sampling.png`),
      });
    await view(page, "Token Journey");
    await view(page, "Sampling Lab");
    await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
      "0.7",
    );
    await page
      .getByRole("button", { name: "Reset defaults", exact: true })
      .click();
    await expect(page.locator(".sampling-applied")).toContainText(
      "temperature 1 · top-k 0 · top-p 1",
    );
    const samplingAxe = await new AxeBuilder({ page }).analyze();
    expect(samplingAxe.violations).toEqual([]);
    await page.setViewportSize({ width: 800, height: 1000 });
    await view(page, "Representation Evolution");
    await expect(page.getByTestId("depth-analysis")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBe(true);
    await page
      .locator(".inspector-panel")
      .screenshot({
        path: resolve(output, `stage6-${country.toLowerCase()}-narrow.png`),
      });
    const after = (await (await request.get(api + "/traces/status")).json())
      .request_forward_passes;
    expect(after).toBe(before + 1);
    expect(errors).toEqual([]);
    await Promise.all(reads);
    save(`stage6-${country.toLowerCase()}-analysis.json`, {
      trace,
      before,
      after,
      jobs,
      accessibility: {
        violations: accessibility.violations,
        passes: accessibility.passes.length,
      },
      samplingAccessibility: {
        violations: samplingAxe.violations,
        passes: samplingAxe.passes.length,
      },
      errors,
    });
    save(`stage6-${country.toLowerCase()}-http-fixtures.json`, fixtures);
  });
