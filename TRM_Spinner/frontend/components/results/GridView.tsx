"use client";

const PALETTE = [
  "#ffffff", // 0 — empty
  "#d4a574", // 1
  "#4a7a4c", // 2
  "#7fb069", // 3
  "#2c4a2e", // 4
  "#f4ca7a", // 5
  "#8d4f30", // 6
  "#b5651d", // 7
  "#1f3a1f", // 8
  "#c8553d", // 9
];

function colorFor(v: number): string {
  if (v < 0) return "#f5f0e8";
  if (v >= PALETTE.length) return `hsl(${(v * 37) % 360} 50% 60%)`;
  return PALETTE[v];
}

interface GridViewProps {
  grid: number[][];
  reference?: number[][]; // for highlighting diffs
  cellSize?: number;
}

/**
 * Render a 2D numeric grid as a small coloured block with values.
 * When `reference` is supplied, cells that differ get a red ring.
 */
export default function GridView({
  grid,
  reference,
  cellSize = 28,
}: GridViewProps) {
  if (!grid || grid.length === 0 || !grid[0]) {
    return (
      <div className="font-mono text-xs text-gator-500/50">(empty)</div>
    );
  }
  const rows = grid.length;
  const cols = Math.max(...grid.map((r) => r.length));

  return (
    <div
      className="inline-grid rounded-md border border-gator-200 bg-cream p-1"
      style={{
        gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
        gap: 2,
      }}
    >
      {grid.flatMap((row, r) =>
        Array.from({ length: cols }, (_, c) => {
          const v = row[c] ?? 0;
          const ref = reference?.[r]?.[c];
          const diff = reference !== undefined && ref !== undefined && ref !== v;
          return (
            <div
              key={`${r}-${c}`}
              className={`flex items-center justify-center rounded-sm font-mono text-[11px] ${
                diff ? "ring-2 ring-red-500" : ""
              }`}
              style={{
                width: cellSize,
                height: cellSize,
                backgroundColor: colorFor(v),
                color: v <= 1 || v === 3 || v === 5 ? "#2c4a2e" : "#ffffff",
              }}
              title={diff && reference ? `expected ${ref}, got ${v}` : String(v)}
            >
              {v}
            </div>
          );
        })
      )}
    </div>
  );
}
