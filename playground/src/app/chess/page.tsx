// Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
// Changes for D1A Copyright 2026 John Soliva: rebranded the UI to D1A (title and header text, the /d1a API proxy, the d1a-latest model name).
import type { Metadata } from "next";
import { ChessGame } from "@/components/chess-game";

export const metadata: Metadata = { title: "D1A · chess", description: "A decision model playing chess: every move is a Choice question." };

export default function ChessPage() {
  return <ChessGame />;
}
