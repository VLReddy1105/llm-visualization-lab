import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import type { MLPInternalsResult, ResidualStreamResult } from "@/lib/api/client";
import { BlockFlow } from "@/components/residual/BlockFlow";
import { OperationDetail, Reconciliation, ResidualDiagnostics } from "@/components/residual/BlockDiagnostics";
import { createSelectionStore } from "@/state/selection";

// Independent real CPU audit results are fixtures only; production never imports them.
const evidence = JSON.parse(readFileSync(resolve(process.cwd(), "../../docs/validation/stage7-scientific-cpu.json"), "utf8")) as {
  prompts: { analyses: { jobs: { result: ResidualStreamResult | MLPInternalsResult }[] } }[];
};
const residual = evidence.prompts[0].analyses.jobs.find((j) => j.result.kind === "residual_stream" && j.result.layer_index === 12 && j.result.source.token.token_index === 4)!.result as ResidualStreamResult;
const mlp = evidence.prompts[0].analyses.jobs.find((j) => j.result.kind === "mlp_internals" && j.result.layer_index === 12 && j.result.source.token.token_index === 4)!.result as MLPInternalsResult;

it("follows keyboard residual operations using real captured norms", async () => {
  function Flow() {
    const [selected, select] = useState("r1");
    return <BlockFlow operations={residual.operations} view="residual" selected={selected} onSelect={select} />;
  }
  render(<Flow />);
  const toolbar = screen.getByRole("toolbar", { name: "Select a computation operation" });
  const buttons = within(toolbar).getAllByRole("button");
  expect(buttons).toHaveLength(8);
  buttons[3].focus();
  await userEvent.keyboard("{ArrowRight}");
  expect(buttons[4]).toHaveFocus();
  expect(buttons[4]).toHaveAttribute("aria-pressed", "true");
  await userEvent.keyboard("{End}");
  expect(buttons[7]).toHaveFocus();
  expect(buttons[7]).toHaveAttribute("aria-pressed", "true");
  expect(buttons[7]).not.toHaveTextContent("Not captured");
});

it("shows the actual SiLU formula and intermediate versus hidden widths", () => {
  const silu = mlp.operations.find((o) => o.symbol === "s")!;
  const { rerender } = render(<OperationDetail operation={silu} />);
  expect(screen.getByText("s = SiLU(g) = g * sigmoid(g)")).toBeVisible();
  expect(screen.getByText("3072", { exact: true })).toBeVisible();
  expect(screen.getByText(/Directly captured/)).toBeVisible();
  rerender(<OperationDetail operation={mlp.operations.find((o) => o.symbol === "m")!} />);
  expect(screen.getByText(/this vector was produced by applying the model’s down projection/)).toBeVisible();
  expect(screen.getByText("1 × 5 × 1024")).toBeVisible();
});

it("keeps undefined zero-vector diagnostics explicit", () => {
  // Isolated display edge case; these substitutions are never production activations.
  const comparison = { ...residual.comparisons[0], before_norm: 0, cosine_before_after: null, update_to_residual_ratio: null,
    undefined: { cosine_before_after: "zero vector norm", update_to_residual_ratio: "zero incoming residual norm" } };
  render(<ResidualDiagnostics comparison={comparison} />);
  expect(screen.getAllByText("Undefined", { exact: true })).toHaveLength(2);
  expect(screen.getByText(/cosine_before_after: undefined — zero vector norm/)).toBeVisible();
  expect(screen.getByText(/update_to_residual_ratio: undefined — zero incoming residual norm/)).toBeVisible();
});

it("distinguishes derived reconciliation from independent module references", () => {
  render(<Reconciliation checks={residual.reconciliation} />);
  expect(screen.getAllByText(/Independent module reference ·/)).toHaveLength(5);
  expect(screen.getAllByText(/Derived from saved operands ·/)).toHaveLength(5);
  expect(screen.getAllByText("PASS", { exact: true })).toHaveLength(10);
});

it("preserves input token, layer and Stage 6 candidate/settings across block views", () => {
  const store = createSelectionStore();
  store.getState().resetTrace("original");
  store.getState().selectToken(4);
  store.getState().selectLayer(12, "source-qualified");
  store.getState().pinCandidate(19846);
  store.getState().setSamplingSettings({ mode: "sample", temperature: .7, top_k: 20, top_p: .8, display_top_k: 10 });
  store.getState().openBlock("mlp", 0);
  store.getState().selectToken(0);
  store.getState().selectLayer(27);
  expect(store.getState()).toMatchObject({ selectedComponent: "residual", selectedToken: 0, selectedLayer: 27, activeVisualizationMode: "residual-mlp" });
  store.getState().openBlock("residual", 4);
  expect(store.getState().selectedToken).toBe(0);
  expect(store.getState().pinnedCandidates).toEqual([19846]);
  expect(store.getState().samplingSettings.temperature).toBe(.7);
  store.getState().resetTrace("new-root");
  expect(store.getState().blockTraces).toEqual({});
  expect(store.getState().selectedToken).toBeNull();
});
