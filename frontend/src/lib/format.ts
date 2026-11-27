export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`)
export const fmtSize = (s: { width: number; height: number }) => `${s.width}×${s.height}`
export const fmtMB = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`
export const fmtParams = (n: number) => `${(n / 1e6).toFixed(2)} M`
export const fmtTime = (t: number) =>
  new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })

/** Colour + label per image source, used identically across charts, captions and badges. */
export const METHOD = {
  input: { label: "Original", color: "var(--c-input)" },
  bicubic: { label: "Bicubic ×4", color: "var(--c-bicubic)" },
  edsr: { label: "EDSR-lite ×4", color: "var(--c-edsr)" },
  ground_truth: { label: "Ground truth", color: "var(--c-gt)" },
} as const
export type MethodKey = keyof typeof METHOD
