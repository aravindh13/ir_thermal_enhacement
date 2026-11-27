// Typed client for the FastAPI backend (backend/app.py). The server is the authority on validation.

export type Mode = "enhance" | "benchmark"
export type Png = { png8: string; png16?: string }
export type Metric = { psnr_db: number; ssim: number }
export type Size = { width: number; height: number }

export type Result = {
  mode: Mode
  scale: number
  device: string
  filename: string
  source: {
    format: string
    mode: string
    bit_depth: number
    value_range: [number, number] | null
    normalised: boolean
    converted_to_grayscale: boolean
  }
  notes: string[]
  input: Size
  output: Size
  images: { input: Png; bicubic: Png; edsr: Png; ground_truth?: Png }
  metrics: null | { reference: string; bicubic: Metric; edsr: Metric }
  timing_ms: Record<string, number>
}

export type Health = { status: string; model_loaded: boolean; device: string; torch_version: string }

export type ModelCard = {
  name: string
  scale: number
  methods: { id: string; label: string; role: string }[]
  parameters: number
  weights_bytes: number
  architecture: Record<string, string | number>
  training: Record<string, string | number>
  dataset: Record<string, string | number>
  reported_results: { source: string; bicubic: Metric; edsr: Metric }
  limits: Record<Mode, { long_side: number; short_side: number }>
  limitations: string[]
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r.json()
}

export const getHealth = () => getJson<Health>("/api/health")
export const getModel = () => getJson<ModelCard>("/api/model")

/** XHR instead of fetch so upload progress is real, not simulated. */
export function process(mode: Mode, file: File, onUpload: (fraction: number) => void): Promise<Result> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open("POST", `/api/${mode}`)
    xhr.responseType = "json"
    xhr.upload.onprogress = (e) => e.lengthComputable && onUpload(e.loaded / e.total)
    xhr.onload = () => {
      if (xhr.status === 200) return resolve(xhr.response as Result)
      const detail = xhr.response?.detail
      reject(new Error(typeof detail === "string" ? detail : `Request failed (HTTP ${xhr.status}).`))
    }
    xhr.onerror = () => reject(new Error("Cannot reach the backend. Is `uvicorn backend.app:app` running?"))
    const body = new FormData()
    body.append("file", file)
    xhr.send(body)
  })
}

export const ACCEPT = ".png,.jpg,.jpeg,.tif,.tiff"
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024
