import {
  test,
  expect,
  type Page,
  type APIRequestContext,
} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type {
  TraceEnvelope,
  TraceStoreStatus,
  AttentionRelation,
  TensorSlice,
} from "../../src/lib/api/client";

const apiUrl = "http://127.0.0.1:8000/api/v1";
const output = resolve(
  process.env.LLM_LAB_BROWSER_OUTPUT_DIR ?? "../../docs/validation",
);
const save = (name: string, data: unknown) =>
  writeFileSync(resolve(output, name), JSON.stringify(data, null, 2) + "\n");
const status = async (request: APIRequestContext): Promise<TraceStoreStatus> =>
  (await request.get(`${apiUrl}/traces/status`)).json();
async function runOverview(page: Page, prompt: string) {
  await page.getByLabel("Raw prompt").fill(prompt);
  const ready = page.waitForResponse(async (response) => {
    if (
      response.request().method() !== "GET" ||
      !/\/traces\/[^/]+\?/.test(response.url())
    )
      return false;
    const body: TraceEnvelope = await response.json();
    return body.status === "ready";
  });
  await page.getByRole("button", { name: "TRACE MODEL" }).click();
  const envelope: TraceEnvelope = await (await ready).json();
  await expect(page.getByText("Trace ready", { exact: true })).toBeVisible();
  return envelope;
}
async function captureLayer(page: Page, layer = 12) {
  await page.getByLabel("Jump to layer").selectOption(String(layer));
  await page.locator(".microscope-node.attention-node").click();
  await expect(
    page.getByText("Detailed attention data was not captured in this trace.", {
      exact: true,
    }),
  ).toBeVisible();
  const capture = page.waitForResponse(
    (r) => r.request().method() === "POST" && /\/traces\?/.test(r.url()),
  );
  await page
    .getByRole("button", {
      name: `Capture Layer ${layer} Attention`,
      exact: true,
    })
    .click();
  const response = await capture;
  expect(response.request().postDataJSON().capture).toEqual({
    profile: "attention",
    layers: [layer],
  });
  const envelope: TraceEnvelope = await response.json();
  await expect(page.getByRole("grid")).toBeVisible();
  await expect(page.getByText(/^Outputs match/)).toBeVisible();
  return envelope.trace_id;
}

test("real Qwen attention: Germany, France, all heads, relation fidelity, reuse and zero forwards", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const fixtures: {
    method: string;
    path: string;
    body: unknown;
    response: unknown;
  }[] = [];
  const pending: Promise<void>[] = [];
  page.on("response", (response) => {
    if (!response.url().startsWith(apiUrl) || response.status() >= 400) return;
    pending.push(
      (async () => {
        const req = response.request();
        const url = new URL(response.url());
        const data = await response.json();
        fixtures.push({
          method: req.method(),
          path: url.pathname + url.search,
          body: req.postDataJSON(),
          response: data,
        });
      })(),
    );
  });
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible({
    timeout: 60000,
  });
  const before = await status(request);
  const overview = await runOverview(page, "The capital of Germany is");
  await page
    .getByRole("button", { name: 'Token 3: " Germany"', exact: true })
    .click();
  await page.getByLabel("Jump to layer").selectOption("12");
  await page
    .getByRole("button", { name: "Detailed view", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: /q_norm \/ k_norm/ }),
  ).toBeVisible();
  await expect(
    page.getByText(/V bypasses Q\/K normalization and RoPE/),
  ).toBeVisible();
  await page
    .locator(".inspector-panel")
    .screenshot({ path: resolve(output, "stage5-layer-microscope.png") });
  await page.locator(".microscope-node.attention-node").click();
  expect((await status(request)).request_forward_passes).toBe(
    before.request_forward_passes + 1,
  );
  await expect(
    page.getByRole("button", {
      name: "Capture Layer 12 Attention",
      exact: true,
    }),
  ).toBeVisible();
  const responsePromise = page.waitForResponse(
    (r) => r.request().method() === "POST" && /\/traces\?/.test(r.url()),
  );
  await page
    .getByRole("button", { name: "Capture Layer 12 Attention", exact: true })
    .click();
  const created: TraceEnvelope = await (await responsePromise).json();
  const id = created.trace_id;
  expect(id).not.toBe(overview.trace_id);
  await expect(page.getByRole("grid")).toBeVisible();
  await expect(page.getByLabel("Query token", { exact: true })).toHaveValue(
    "3",
  );
  await expect(page.getByText(/^Outputs match/)).toBeVisible();
  const afterCapture = await status(request);
  expect(afterCapture.request_forward_passes).toBe(
    before.request_forward_passes + 2,
  );
  await page.getByLabel("Query token", { exact: true }).selectOption("4");
  await page.getByRole("button", { name: "Query Head 4", exact: true }).click();
  await expect(
    page.getByText("Query Head 4 → KV Head 2", { exact: true }),
  ).toBeVisible();
  await page.locator('.matrix-cell[data-query="4"][data-key="3"]').click();
  const relation: AttentionRelation = await (
    await request.get(
      `${apiUrl}/traces/${id}/layers/12/attention/heads/4/relation?query_token_index=4&key_token_index=3`,
    )
  ).json();
  const weight = String(relation.observations.weights_reconstructed.values[0]);
  await expect(page.getByTestId("relation-weight")).toHaveText(weight);
  expect(
    await page
      .locator('.matrix-cell[data-query="4"][data-key="3"]')
      .getAttribute("data-weight"),
  ).toBe(weight);
  await expect(page.locator('.matrix-cell[data-masked="true"]')).toHaveCount(
    10,
  );
  await page
    .locator('.matrix-cell[data-query="3"][data-key="4"]')
    .click({ force: true });
  await expect(page.getByLabel("Query token", { exact: true })).toHaveValue(
    "4",
  );
  await expect(page.getByLabel("Key token", { exact: true })).toHaveValue("3");
  const cell = page.locator('.matrix-cell[data-query="4"][data-key="3"]');
  await cell.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(
    page.locator('.matrix-cell[data-query="4"][data-key="2"]'),
  ).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Key token", { exact: true })).toHaveValue("2");
  await cell.click();
  await page
    .locator(".attention-explorer")
    .screenshot({ path: resolve(output, "stage5-attention-heatmap.png") });
  await page
    .locator(".attention-reading")
    .screenshot({ path: resolve(output, "stage5-token-relation.png") });
  await page
    .getByRole("button", { name: "Inspect Q / K / V vectors", exact: true })
    .click();
  await expect(page.getByTestId("q-l2_norm")).toBeVisible();
  await expect(page.getByTestId("k-l2_norm")).toBeVisible();
  await expect(page.getByTestId("v-l2_norm")).toBeVisible();
  const norms = {
    q: await page.getByTestId("q-l2_norm").textContent(),
    k: await page.getByTestId("k-l2_norm").textContent(),
    v: await page.getByTestId("v-l2_norm").textContent(),
  };
  for (const [name, suffix, headAxis, tokenAxis, h, t] of [
    ["q", "query.rotary", "query_head", "query_token", 4, 4],
    ["k", "key.rotary", "kv_head", "key_token", 2, 3],
    ["v", "value", "kv_head", "key_token", 2, 3],
  ] as const) {
    const vector: TensorSlice = await (
      await request.post(
        `${apiUrl}/traces/${id}/components/layer[12].attention.${suffix}/slice`,
        {
          data: {
            selectors: {
              batch: { index: 0 },
              [headAxis]: { index: h },
              [tokenAxis]: { index: t },
            },
            include_statistics: true,
          },
        },
      )
    ).json();
    expect(norms[name]).toBe(String(vector.statistics!.l2_norm));
    expect(vector.values).toHaveLength(128);
  }
  await page
    .locator(".relation-inspector")
    .screenshot({ path: resolve(output, "stage5-qkv-inspector.png") });
  await page
    .getByText("Show raw values · Q after RoPE", { exact: true })
    .click();
  await expect(
    page.locator(".vector-inspector").first().locator("pre"),
  ).toContainText('"provenance": "reconstructed"');
  await page
    .getByText("Show raw values · Q after RoPE", { exact: true })
    .click();
  for (const stage of ["projected", "normalized", "rotary"]) {
    await page.getByLabel("Q / K observation stage").selectOption(stage);
    await expect(page.getByTestId("q-l2_norm")).toBeVisible();
  }
  const accessibility = await new AxeBuilder({ page }).analyze();
  save("stage5-accessibility.json", {
    violations: accessibility.violations,
    passes: accessibility.passes.length,
  });
  expect(accessibility.violations).toEqual([]);
  await page
    .getByRole("button", { name: "Hide Q / K / V vectors", exact: true })
    .click();
  const headMappings = [];
  for (let head = 0; head < 16; head++) {
    await page
      .getByRole("button", { name: `Query Head ${head}`, exact: true })
      .click();
    const metadata = await (
      await request.get(
        `${apiUrl}/traces/${id}/layers/12/attention/heads/${head}`,
      )
    ).json();
    await expect(
      page.getByText(`Query Head ${head} → KV Head ${metadata.kv_head_index}`, {
        exact: true,
      }),
    ).toBeVisible();
    const expected = await (
      await request.get(
        `${apiUrl}/traces/${id}/layers/12/attention/heads/${head}/relation?query_token_index=4&key_token_index=3`,
      )
    ).json();
    await expect(page.getByTestId("relation-weight")).toHaveText(
      String(expected.observations.weights_reconstructed.values[0]),
    );
    headMappings.push([head, metadata.kv_head_index]);
  }
  await page.getByRole("button", { name: "Query Head 4", exact: true }).click();
  await page
    .locator(".attention-explorer")
    .screenshot({ path: resolve(output, "stage5-attention-head.png") });
  await page
    .getByRole("button", { name: 'Token 3: " Germany"', exact: true })
    .click();
  await expect(page.getByRole("grid")).toBeVisible();
  await expect(page.getByLabel("Query token", { exact: true })).toHaveValue(
    "3",
  );
  await expect(page.locator(".connection-keys button")).toHaveCount(4);
  await page.getByLabel("Key token", { exact: true }).selectOption("4");
  await expect(page.getByText(/Causally masked · this key/)).toBeVisible();
  await expect(page.getByTestId("relation-weight")).toHaveCount(0);
  await page.getByLabel("Key token", { exact: true }).selectOption("3");
  await page
    .getByRole("button", { name: "← Back to Layer", exact: true })
    .click();
  await page.locator(".microscope-node.attention-node").click();
  await expect(page.getByRole("grid")).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Capture Layer 12 Attention",
      exact: true,
    }),
  ).toHaveCount(0);
  expect((await status(request)).request_forward_passes).toBe(
    afterCapture.request_forward_passes,
  );
  await page.getByLabel("Jump to layer").selectOption("13");
  await page.locator(".microscope-node.attention-node").click();
  await expect(
    page.getByRole("button", {
      name: "Capture Layer 13 Attention",
      exact: true,
    }),
  ).toBeVisible();
  expect((await status(request)).request_forward_passes).toBe(
    afterCapture.request_forward_passes,
  );
  const germany = {
    overviewId: overview.trace_id,
    deepId: id,
    device: overview.overview!.model.device,
    relation,
    norms,
    headMappings,
    forwardsBefore: before.request_forward_passes,
    forwardsAfterCapture: afterCapture.request_forward_passes,
    forwardsAfterExploration: (await status(request)).request_forward_passes,
  };
  save("stage5-germany-browser.json", germany);
  await Promise.all(pending);
  save("stage5-http-fixtures.json", fixtures);
  const franceOverview = await runOverview(page, "The capital of France is");
  const franceId = await captureLayer(page);
  await page.getByRole("button", { name: "Query Head 4", exact: true }).click();
  await page.locator('.matrix-cell[data-query="4"][data-key="3"]').click();
  const franceRelation: AttentionRelation = await (
    await request.get(
      `${apiUrl}/traces/${franceId}/layers/12/attention/heads/4/relation?query_token_index=4&key_token_index=3`,
    )
  ).json();
  await expect(page.getByTestId("relation-weight")).toHaveText(
    String(franceRelation.observations.weights_reconstructed.values[0]),
  );
  await page
    .getByRole("button", { name: "Inspect Q / K / V vectors", exact: true })
    .click();
  await expect(page.getByTestId("v-l2_norm")).toBeVisible();
  save("stage5-france-browser.json", {
    overviewId: franceOverview.trace_id,
    deepId: franceId,
    relation: franceRelation,
    norms: {
      q: await page.getByTestId("q-l2_norm").textContent(),
      k: await page.getByTestId("k-l2_norm").textContent(),
      v: await page.getByTestId("v-l2_norm").textContent(),
    },
    forwards: (await status(request)).request_forward_passes,
    browserErrors: errors,
  });
  await page
    .locator(".attention-explorer")
    .screenshot({ path: resolve(output, "stage5-france-attention.png") });
  expect(errors).toEqual([]);
  expect((await status(request)).request_forward_passes).toBe(
    before.request_forward_passes + 4,
  );
});

test("real longer sequence uses bounded windows and retains full-row normalization", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible();
  const overview = await runOverview(
    page,
    "The capital of Germany is Berlin. The capital of France is Paris. The capital of Italy is Rome. These cities are located in Europe.",
  );
  const id = await captureLayer(page);
  const count = overview.overview!.input.sequence_length;
  expect(count).toBeGreaterThan(16);
  expect(await page.locator(".matrix-cell").count()).toBeLessThanOrEqual(256);
  await page
    .getByRole("button", { name: "Previous keys", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Previous queries", exact: true })
    .click();
  await expect(
    page.locator('.matrix-cell[data-query="0"][data-key="0"]'),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Center selection", exact: true })
    .click();
  await expect(
    page.locator(
      `.matrix-cell[data-query="${count - 1}"][data-key="${count - 1}"]`,
    ),
  ).toBeVisible();
  expect(await page.locator(".matrix-cell").count()).toBeLessThanOrEqual(256);
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  await page.setViewportSize({ width: 800, height: 1100 });
  await expect
    .poll(() =>
      page.locator(".heatmap-scroll").evaluate((container) => {
        const cell = container.querySelector(".selected-cell")!;
        const a = container.getBoundingClientRect(),
          b = cell.getBoundingClientRect();
        return (
          b.left >= a.left &&
          b.right <= a.right &&
          b.top >= a.top &&
          b.bottom <= a.bottom
        );
      }),
    )
    .toBe(true);

  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page
    .locator(".attention-explorer")
    .screenshot({ path: resolve(output, "stage5-long-sequence.png") });
  save("stage5-long-sequence.json", {
    traceId: id,
    tokenCount: count,
    renderedCells: await page.locator(".matrix-cell").count(),
    accessibilityViolations: accessibility.violations,
  });
  expect((await request.get(`${apiUrl}/traces/${id}`)).status()).toBe(200);
});
