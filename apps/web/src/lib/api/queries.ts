import { queryOptions } from "@tanstack/react-query";
import { api, type TokenMetadata } from "./client";

export const traceKey = (id: string | null) => ["trace", id] as const;
export const overviewQuery = (id: string | null) =>
  queryOptions({
    queryKey: [...traceKey(id), "overview"],
    queryFn: ({ signal }) => api.trace(id!, 10, signal),
    enabled: !!id,
    refetchInterval: (query) =>
      !query.state.error &&
      ["queued", "running"].includes(query.state.data?.status ?? "")
        ? 500
        : false,
  });
export const tokensQuery = (id: string) =>
  queryOptions({
    queryKey: [...traceKey(id), "tokens"],
    queryFn: async ({ signal }) => {
      const tokens: TokenMetadata[] = [];
      let cursor: string | undefined;
      do {
        const page = await api.traceTokens(id, { limit: 200, cursor }, signal);
        tokens.push(...page.items);
        cursor = page.next_cursor ?? undefined;
      } while (cursor);
      return tokens;
    },
  });
export const layersQuery = (id: string) =>
  queryOptions({
    queryKey: [...traceKey(id), "layers"],
    queryFn: ({ signal }) => api.traceLayers(id, { limit: 200 }, signal),
  });
export const layerQuery = (id: string, layer: number) =>
  queryOptions({
    queryKey: [...traceKey(id), "layer", layer],
    queryFn: ({ signal }) => api.traceLayer(id, layer, signal),
  });
export const componentQuery = (id: string, component: string) =>
  queryOptions({
    queryKey: [...traceKey(id), "component", component],
    queryFn: ({ signal }) => api.traceComponent(id, component, signal),
  });
export const logitsQuery = (id: string) =>
  queryOptions({
    queryKey: [...traceKey(id), "logits"],
    queryFn: ({ signal }) => api.traceLogits(id, 10, undefined, signal),
  });
