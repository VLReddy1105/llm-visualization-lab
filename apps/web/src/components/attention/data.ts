import { queryOptions } from "@tanstack/react-query";
import { api, type SliceRequest, type TokenMetadata } from "@/lib/api/client";
import { traceKey } from "@/lib/api/queries";

export const sliceQuery = (
  id: string,
  component: string,
  selectors: SliceRequest["selectors"],
  statistics = false,
) =>
  queryOptions({
    queryKey: [...traceKey(id), "slice", component, selectors, statistics],
    queryFn: ({ signal }) =>
      api.traceSlice(
        id,
        component,
        { selectors, include_statistics: statistics },
        signal,
      ),
  });
export const tokenDescription = (token: TokenMetadata) =>
  `Position ${token.token_index}; decoded ${JSON.stringify(token.decoded_token)}; native ${JSON.stringify(token.tokenizer_piece)}; token ID ${token.token_id}`;
export const exact = (value: number | null | undefined) =>
  value == null ? "Unavailable" : String(value);
export const isMasked = (value: number | null | undefined) =>
  value == null || value < 0;
export const attentionComponent = (layer: number, suffix: string) =>
  `layer[${layer}].attention.${suffix}`;
