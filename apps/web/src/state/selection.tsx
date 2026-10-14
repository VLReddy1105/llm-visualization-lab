"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { createStore } from "zustand/vanilla";
import { useStore } from "zustand";
import type { ApiError, SamplingParameters } from "@/lib/api/client";

export type JourneyComponent =
  | "residual"
  | "evolution"
  | "input"
  | "tokens"
  | "embedding.output"
  | "layer"
  | "attention"
  | "output.normalized"
  | "output.lm_head_input"
  | "logits"
  | "prediction"
  | "token";
export type EvolutionView =
  "journey" | "representation" | "lens" | "candidates" | "sampling";
export type SelectionState = {
  blockView: "residual" | "mlp";
  blockTraces: Record<string, string>;
  openBlock: (view: "residual" | "mlp", fallbackToken: number, layer?: number) => void;
  associateBlock: (key: string, id: string) => void;
  samplingSettings: Omit<SamplingParameters, "token_index">;
  setSamplingSettings: (
    settings: Omit<SamplingParameters, "token_index">,
  ) => void;
  evolutionView: EvolutionView;
  selectedDepth: number | null;
  microscopeTraceId: string | null;
  pinnedCandidates: number[];
  evolutionTraces: Record<string, string>;
  openEvolution: (view: EvolutionView, fallbackToken: number) => void;
  selectDepth: (depth: number, layer: number | null) => void;
  pinCandidate: (id: number) => void;
  associateEvolution: (key: string, id: string) => void;
  traceId: string | null;
  selectedToken: number | null;
  selectedLayer: number | null;
  selectedComponent: JourneyComponent;
  selectedHead: number | null;
  selectedQueryToken: number | null;
  selectedKeyToken: number | null;
  selectedOperation: string | null;
  attentionTraces: Record<string, string>;
  detailIssues: Record<string, ApiError>;
  selectedCandidateToken: number | null;
  selectedGenerationStep: number | null;
  activeVisualizationMode:
    "model-journey" | "layer-microscope" | "attention" | "token-evolution" | "residual-mlp";
  viewport: "comfortable" | "compact";
  traceIssue: ApiError | null;
  resetTrace: (id: string | null) => void;
  selectToken: (index: number) => void;
  selectLayer: (index: number, sourceId?: string) => void;
  selectAttention: (layer: number, fallbackToken: number) => void;
  selectHead: (index: number) => void;
  selectRelation: (query: number, key: number) => void;
  selectOperation: (operation: string | null) => void;
  associateAttention: (key: string, id: string) => void;
  setDetailIssue: (id: string, error: ApiError) => void;
  selectComponent: (component: JourneyComponent) => void;
  selectCandidate: (id: number) => void;
  setViewport: (mode: SelectionState["viewport"]) => void;
  setTraceIssue: (error: ApiError | null) => void;
};

export const createSelectionStore = () =>
  createStore<SelectionState>()((set) => ({
    blockView: "residual",
    blockTraces: {},
    openBlock: (blockView, fallbackToken, layer) => set((s) => ({
      blockView,
      selectedComponent: "residual",
      activeVisualizationMode: "residual-mlp",
      selectedToken: s.selectedToken ?? fallbackToken,
      selectedLayer: layer ?? s.selectedLayer ?? 0,
      selectedOperation: null,
    })),
    associateBlock: (key, id) => set((s) => ({ blockTraces: { ...s.blockTraces, [key]: id } })),
    samplingSettings: {
      temperature: 1,
      top_k: 0,
      top_p: 1,
      display_top_k: 10,
      mode: "sample",
    },
    setSamplingSettings: (samplingSettings) => set({ samplingSettings }),
    evolutionView: "journey",
    selectedDepth: null,
    microscopeTraceId: null,
    pinnedCandidates: [],
    evolutionTraces: {},
    traceId: null,
    selectedToken: null,
    selectedLayer: null,
    selectedComponent: "input",
    selectedHead: null,
    selectedQueryToken: null,
    selectedKeyToken: null,
    selectedOperation: null,
    attentionTraces: {},
    detailIssues: {},
    selectedCandidateToken: null,
    selectedGenerationStep: null,
    activeVisualizationMode: "model-journey",
    viewport: "comfortable",
    traceIssue: null,
    resetTrace: (traceId) =>
      set({
        traceId,
        blockView: "residual",
        blockTraces: {},
        samplingSettings: {
          temperature: 1,
          top_k: 0,
          top_p: 1,
          display_top_k: 10,
          mode: "sample",
        },
        evolutionView: "journey",
        selectedDepth: null,
        microscopeTraceId: null,
        pinnedCandidates: [],
        evolutionTraces: {},
        selectedToken: null,
        selectedLayer: null,
        selectedComponent: "input",
        selectedHead: null,
        selectedQueryToken: null,
        selectedKeyToken: null,
        selectedOperation: null,
        attentionTraces: {},
        detailIssues: {},
        activeVisualizationMode: "model-journey",
        selectedCandidateToken: null,
        selectedGenerationStep: null,
        traceIssue: null,
        viewport: "comfortable",
      }),
    selectToken: (selectedToken) =>
      set((state) =>
        state.selectedComponent === "evolution" || state.selectedComponent === "residual"
          ? { selectedToken }
          : state.selectedComponent === "attention"
            ? {
                selectedToken,
                selectedQueryToken: selectedToken,
                selectedKeyToken: selectedToken,
              }
            : {
                selectedToken,
                selectedComponent: "token",
                activeVisualizationMode: "model-journey",
              },
      ),
    selectLayer: (selectedLayer, sourceId) =>
      set((state) => ({
        selectedLayer,
        selectedDepth: selectedLayer + 1,
        microscopeTraceId: sourceId ?? null,
        selectedComponent: state.selectedComponent === "residual" ? "residual" : "layer",
        selectedHead:
          state.selectedLayer === selectedLayer ? state.selectedHead : null,
        selectedQueryToken:
          state.selectedLayer === selectedLayer
            ? state.selectedQueryToken
            : null,
        selectedKeyToken:
          state.selectedLayer === selectedLayer ? state.selectedKeyToken : null,
        selectedOperation: null,
        activeVisualizationMode: state.selectedComponent === "residual" ? "residual-mlp" : "layer-microscope",
      })),
    selectAttention: (selectedLayer, fallbackToken) =>
      set((state) => ({
        selectedLayer,
        selectedComponent: "attention",
        activeVisualizationMode: "attention",
        selectedHead: state.selectedHead ?? 0,
        selectedQueryToken: state.selectedToken ?? fallbackToken,
        selectedKeyToken:
          state.selectedLayer === selectedLayer &&
          state.selectedQueryToken === state.selectedToken
            ? (state.selectedKeyToken ?? state.selectedToken ?? fallbackToken)
            : (state.selectedToken ?? fallbackToken),
        selectedToken: state.selectedToken ?? fallbackToken,
      })),
    openEvolution: (evolutionView, fallbackToken) =>
      set((s) => ({
        evolutionView,
        selectedComponent: "evolution",
        activeVisualizationMode: "token-evolution",
        selectedToken: s.selectedToken ?? fallbackToken,
      })),
    selectDepth: (selectedDepth, selectedLayer) =>
      set({ selectedDepth, selectedLayer }),
    pinCandidate: (id) =>
      set((s) => ({
        pinnedCandidates: s.pinnedCandidates.includes(id)
          ? s.pinnedCandidates.filter((candidate) => candidate !== id)
          : s.pinnedCandidates.length < 8
            ? [...s.pinnedCandidates, id].sort((a, b) => a - b)
            : s.pinnedCandidates,
      })),
    associateEvolution: (key, id) =>
      set((s) => ({ evolutionTraces: { ...s.evolutionTraces, [key]: id } })),
    selectHead: (selectedHead) => set({ selectedHead }),
    selectRelation: (selectedQueryToken, selectedKeyToken) =>
      set({
        selectedQueryToken,
        selectedKeyToken,
        selectedToken: selectedQueryToken,
      }),
    selectOperation: (selectedOperation) => set({ selectedOperation }),
    associateAttention: (key, id) =>
      set((state) => ({
        attentionTraces: { ...state.attentionTraces, [key]: id },
      })),
    setDetailIssue: (id, error) =>
      set((state) => ({
        detailIssues: { ...state.detailIssues, [id]: error },
      })),
    selectComponent: (selectedComponent) =>
      set({
        selectedComponent,
        activeVisualizationMode:
          selectedComponent === "attention"
            ? "attention"
            : selectedComponent === "layer"
              ? "layer-microscope"
              : "model-journey",
      }),
    selectCandidate: (selectedCandidateToken) =>
      set({ selectedCandidateToken }),
    setViewport: (viewport) => set({ viewport }),
    setTraceIssue: (traceIssue) => set({ traceIssue }),
  }));

const SelectionContext = createContext<ReturnType<
  typeof createSelectionStore
> | null>(null);
export function SelectionProvider({ children }: { children: ReactNode }) {
  const [store] = useState(createSelectionStore);
  return (
    <SelectionContext.Provider value={store}>
      {children}
    </SelectionContext.Provider>
  );
}
export function useSelectionStore() {
  const store = useContext(SelectionContext);
  if (!store) throw new Error("SelectionProvider is required.");
  return store;
}
export function useSelection<T>(selector: (state: SelectionState) => T): T {
  return useStore(useSelectionStore(), selector);
}
