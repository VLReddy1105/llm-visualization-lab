"use client";

import {
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { ApiError } from "@/lib/api/client";
import { SelectionProvider, useSelectionStore } from "@/state/selection";

function ServerStateProvider({ children }: { children: ReactNode }) {
  const selection = useSelectionStore();
  const [client] = useState(
    () =>
      new QueryClient({
        queryCache: new QueryCache({
          onError: (error, query) => {
            const state = selection.getState();
            if (
              query.queryKey[0] === "trace" &&
              query.queryKey[1] === state.traceId &&
              error instanceof ApiError &&
              ["TRACE_EXPIRED", "TRACE_EVICTED", "TRACE_NOT_FOUND"].includes(
                error.code,
              )
            ) {
              state.setTraceIssue(error);
            }
          },
        }),
        defaultOptions: {
          queries: {
            staleTime: Infinity,
            gcTime: 5 * 60 * 1000,
            retry: false,
            refetchOnWindowFocus: false,
            refetchOnReconnect: false,
          },
          mutations: { retry: false },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
export function WorkspaceProviders({ children }: { children: ReactNode }) {
  return (
    <SelectionProvider>
      <ServerStateProvider>{children}</ServerStateProvider>
    </SelectionProvider>
  );
}
