import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Visualization Lab",
  description:
    "Explore one real Qwen3-0.6B computation, from input tokens through transformer layers to next-token predictions.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
