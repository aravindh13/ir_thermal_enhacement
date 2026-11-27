// Recent activity, browser storage only (approved: no server-side history).
import type { Metric, Mode, Result, Size } from "./api"

export type Run = {
  id: string
  time: number
  filename: string
  mode: Mode
  input: Size
  output: Size
  bit_depth: number
  edsr_ms: number
  total_ms: number
  metrics: null | { bicubic: Metric; edsr: Metric }
}

const KEY = "irsr.history.v1"
const MAX = 50

export function loadRuns(): Run[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "[]")
    return Array.isArray(v) ? v : []
  } catch {
    return []
  }
}

export function addRun(r: Result): Run[] {
  const run: Run = {
    id: crypto.randomUUID(),
    time: Date.now(),
    filename: r.filename,
    mode: r.mode,
    input: r.input,
    output: r.output,
    bit_depth: r.source.bit_depth,
    edsr_ms: r.timing_ms.edsr,
    total_ms: r.timing_ms.total,
    metrics: r.metrics && { bicubic: r.metrics.bicubic, edsr: r.metrics.edsr },
  }
  const runs = [run, ...loadRuns()].slice(0, MAX)
  localStorage.setItem(KEY, JSON.stringify(runs))
  return runs
}

export function clearRuns(): Run[] {
  localStorage.removeItem(KEY)
  return []
}
