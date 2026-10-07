import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Visualization Lab",
  description: "Real raw next-token inference from Qwen3-0.6B. Stage 1 foundation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
