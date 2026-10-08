"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { createStore } from "zustand/vanilla";
import { useStore } from "zustand";
import type { ApiError } from "@/lib/api/client";

export type JourneyComponent =
  | "input"
  | "tokens"
  | "embedding.output"
  | "layer"
  | "output.normalized"
  | "output.lm_head_input"
  | "logits"
  | "prediction"
  | "token";
export type SelectionState = {
  traceId: string | null;
  selectedToken: number | null;
  selectedLayer: number | null;
  selectedComponent: JourneyComponent;
  selectedHead: number | null;
  selectedCandidateToken: number | null;
  selectedGenerationStep: number | null;
  activeVisualizationMode: "model-journey";
  viewport: "comfortable" | "compact";
  traceIssue: ApiError | null;
  resetTrace: (id: string | null) => void;
  selectToken: (index: number) => void;
  selectLayer: (index: number) => void;
  selectComponent: (component: JourneyComponent) => void;
  selectCandidate: (id: number) => void;
  setViewport: (mode: SelectionState["viewport"]) => void;
  setTraceIssue: (error: ApiError | null) => void;
};

export const createSelectionStore = () =>
  createStore<SelectionState>()((set) => ({
    traceId: null,
    selectedToken: null,
    selectedLayer: null,
    selectedComponent: "input",
    selectedHead: null,
    selectedCandidateToken: null,
    selectedGenerationStep: null,
    activeVisualizationMode: "model-journey",
    viewport: "comfortable",
    traceIssue: null,
    resetTrace: (traceId) =>
      set({
        traceId,
        selectedToken: null,
        selectedLayer: null,
        selectedComponent: "input",
        selectedHead: null,
        selectedCandidateToken: null,
        selectedGenerationStep: null,
        traceIssue: null,
        viewport: "comfortable",
      }),
    selectToken: (selectedToken) =>
      set({ selectedToken, selectedComponent: "token" }),
    selectLayer: (selectedLayer) =>
      set({ selectedLayer, selectedComponent: "layer", selectedHead: null }),
    selectComponent: (selectedComponent) => set({ selectedComponent }),
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
