import { useEffect, useRef, useState } from "react"
import { Download, GitCompare, Loader2, X } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { Page } from "@/components/AppShell"
import { EmptyState, PageHeader, Section } from "@/components/common"
import { UploadDropzone } from "@/components/UploadDropzone"
import { ACCEPT, MAX_UPLOAD_BYTES, process, type Mode, type ModelCard, type Png, type Result } from "@/lib/api"
import { fmtMs, fmtSize, METHOD, type MethodKey } from "@/lib/format"

type Status =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number; started: number }
  | { kind: "processing"; started: number }
  | { kind: "done" }
  | { kind: "error"; message: string }

const MODE_HELP: Record<Mode, string> = {
  enhance: "Your image is the low-resolution input. Returns bicubic ×4 and EDSR-lite ×4. No PSNR/SSIM: there is no ground truth.",
  benchmark: "Your image is the high-resolution reference. It is bicubic-downscaled ×4 (as in training), reconstructed, and scored with PSNR/SSIM.",
}

/** Client-side pre-check for fast feedback; the backend re-validates everything. */
function precheck(f: File, mode: Mode, card: ModelCard | null, dims: { w: number; h: number } | null): string | null {
  const ext = f.name.split(".").pop()?.toLowerCase() ?? ""
  if (!ACCEPT.split(",").includes(`.${ext}`)) return "Unsupported file type. Use PNG, JPG, JPEG or TIFF."
  if (f.size > MAX_UPLOAD_BYTES) return "File is larger than 20 MB."
  const lim = card?.limits[mode]
  if (lim && dims && (Math.max(dims.w, dims.h) > lim.long_side || Math.min(dims.w, dims.h) > lim.short_side))
    return `Image is ${dims.w}×${dims.h}; ${mode} mode accepts at most ${lim.long_side}×${lim.short_side} (either orientation).`
  return null
}

function base(name: string) {
  return name.replace(/\.[^.]+$/, "")
}

export function Workspace({ card, result, onResult, go }: {
  card: ModelCard | null; result: Result | null; onResult: (r: Result) => void; go: (p: Page) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [mode, setMode] = useState<Mode>("enhance")
  const [status, setStatus] = useState<Status>(result ? { kind: "done" } : { kind: "idle" })
  const [now, setNow] = useState(() => Date.now())
  const tick = useRef<number>(0)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])
  const busy = status.kind === "uploading" || status.kind === "processing"
  useEffect(() => {
    if (!busy) return
    tick.current = window.setInterval(() => setNow(Date.now()), 100)
    return () => clearInterval(tick.current)
  }, [busy])

  const pick = (f: File) => {
    setFile(f)
    setDims(null)
    setPreviewFailed(false)
    setPreview(URL.createObjectURL(f))
    setStatus({ kind: "idle" })
  }

  const run = async () => {
    if (!file) return
    const err = precheck(file, mode, card, dims)
    if (err) return setStatus({ kind: "error", message: err })
    const started = Date.now()
    setStatus({ kind: "uploading", fraction: 0, started })
    try {
      const r = await process(mode, file, (fraction) =>
        setStatus(fraction < 1 ? { kind: "uploading", fraction, started } : { kind: "processing", started }))
      onResult(r)
      setStatus({ kind: "done" })
    } catch (e) {
      setStatus({ kind: "error", message: (e as Error).message })
    }
  }

  const lim = card?.limits[mode]
  const pre = file ? precheck(file, mode, card, dims) : null

  return (
    <>
      <PageHeader title="Enhancement workspace" description="Upload a grayscale thermal image and run bicubic and EDSR-lite ×4 side by side." />

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-6">
          <Section title="1 · Image">
            <UploadDropzone onFile={pick} disabled={busy} />
            {file && (
              <div className="mt-3 flex items-center justify-between gap-2 rounded-lg bg-muted px-3 py-2 text-sm">
                <span className="truncate font-medium">{file.name}</span>
                <span className="num shrink-0 text-xs text-muted-foreground">
                  {dims ? `${dims.w}×${dims.h} · ` : ""}{(file.size / 1024).toFixed(0)} KB
                </span>
                <button aria-label="Remove file" disabled={busy} onClick={() => { setFile(null); setPreview(null); setDims(null) }}
                  className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-40">
                  <X className="size-4" />
                </button>
              </div>
            )}
          </Section>

          <Section title="2 · Mode">
            <ToggleGroup
              variant="outline"
              value={[mode]}
              onValueChange={(v) => v[0] && setMode(v[0] as Mode)}
              className="w-full"
              spacing={0}
              aria-label="Processing mode"
            >
              <ToggleGroupItem value="enhance" className="flex-1" disabled={busy}>Enhance</ToggleGroupItem>
              <ToggleGroupItem value="benchmark" className="flex-1" disabled={busy}>Benchmark</ToggleGroupItem>
            </ToggleGroup>
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{MODE_HELP[mode]}</p>
            {lim && (
              <p className="num mt-2 text-xs text-muted-foreground">
                Size limit: {lim.long_side}×{lim.short_side} {mode === "enhance" ? "input" : "reference"} (either orientation)
              </p>
            )}
          </Section>

          <Section title="3 · Run">
            <p className="mb-3 text-xs text-muted-foreground">Methods: <span className="font-medium text-foreground">Bicubic ×4</span> baseline + <span className="font-medium text-foreground">EDSR-lite ×4</span></p>
            <Button className="w-full" size="lg" disabled={!file || busy || !!pre} onClick={run}>
              {busy && <Loader2 className="animate-spin" data-icon="inline-start" />}
              {busy ? "Processing…" : mode === "enhance" ? "Enhance ×4" : "Run benchmark"}
            </Button>
            {pre && <p role="alert" className="mt-2 text-xs text-destructive">{pre}</p>}
          </Section>
        </div>

        <div className="space-y-6">
          <StatusPanel status={status} now={now} />
          {status.kind !== "done" || !result ? (
            <Section title="Original" description={dims ? `${dims.w}×${dims.h} · ${file?.type || "image"}` : undefined}>
              {!preview ? (
                <EmptyState title="No image selected">Choose a file to preview it here.</EmptyState>
              ) : previewFailed ? (
                <EmptyState title="Preview not available">This browser cannot display TIFF files. The image will still be processed.</EmptyState>
              ) : (
                <img src={preview} alt="Uploaded original" className="pixelated mx-auto max-h-[520px] rounded-lg border bg-zinc-100"
                  onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                  onError={() => setPreviewFailed(true)} />
              )}
            </Section>
          ) : (
            <Results r={result} go={go} />
          )}
        </div>
      </div>
    </>
  )
}

function StatusPanel({ status, now }: { status: Status; now: number }) {
  if (status.kind === "error")
    return (
      <Alert variant="destructive">
        <AlertTitle>Could not process image</AlertTitle>
        <AlertDescription>{status.message}</AlertDescription>
      </Alert>
    )
  if (status.kind === "uploading" || status.kind === "processing") {
    const elapsed = fmtMs(now - status.started)
    return (
      <div role="status" aria-live="polite" className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
        <div className="mb-2 flex justify-between text-sm">
          <span className="font-medium">{status.kind === "uploading" ? "Uploading…" : "Running bicubic and EDSR-lite on the server…"}</span>
          <span className="num text-muted-foreground">{elapsed}</span>
        </div>
        {/* Upload progress is measured; server-side progress is not reported, so that phase is indeterminate. */}
        {status.kind === "uploading" ? (
          <Progress value={Math.round(status.fraction * 100)} aria-label="Upload progress" />
        ) : (
          <div className="h-1 w-full overflow-hidden rounded-full bg-muted" role="progressbar" aria-label="Processing">
            <div className="animate-indeterminate h-full w-1/4 rounded-full bg-primary" />
          </div>
        )}
      </div>
    )
  }
  return null
}

function Downloads({ png, name }: { png: Png; name: string }) {
  return (
    <div className="flex flex-wrap gap-2">
      <a className={buttonVariants({ variant: "outline", size: "sm" })} href={png.png8} download={`${name}.png`}>
        <Download data-icon="inline-start" /> PNG (8-bit)
      </a>
      {png.png16 && (
        <a className={buttonVariants({ variant: "outline", size: "sm" })} href={png.png16} download={`${name}_16bit.png`}>
          <Download data-icon="inline-start" /> PNG (16-bit, sensor units)
        </a>
      )}
    </div>
  )
}

function Results({ r, go }: { r: Result; go: (p: Page) => void }) {
  const keys: MethodKey[] = r.images.ground_truth ? ["input", "bicubic", "edsr", "ground_truth"] : ["input", "bicubic", "edsr"]
  const size = (k: MethodKey) => (k === "input" ? r.input : r.output)
  const name = base(r.filename)
  const inputLabel = r.mode === "benchmark" ? "Synthetic LR input" : METHOD.input.label

  return (
    <Section
      title="Result"
      description={`${r.filename} · ${r.mode} · ${fmtSize(r.input)} → ${fmtSize(r.output)} · ${r.device.toUpperCase()}`}
      action={<Button size="sm" onClick={() => go("compare")}><GitCompare data-icon="inline-start" /> Compare</Button>}
    >
      <div className={`grid gap-4 sm:grid-cols-2 ${keys.length === 4 ? "xl:grid-cols-4" : "xl:grid-cols-3"}`}>
        {keys.map((k) => (
          <figure key={k} className="space-y-2">
            <img src={r.images[k]!.png8} alt={METHOD[k].label} className="pixelated w-full rounded-lg border bg-zinc-100"
              style={{ aspectRatio: r.output.width / r.output.height }} />
            <figcaption className="flex items-center justify-between text-xs">
              <span className="flex items-center gap-1.5 font-medium">
                <span className="size-2 rounded-sm" style={{ background: METHOD[k].color }} aria-hidden />
                {k === "input" ? inputLabel : METHOD[k].label}
              </span>
              <span className="num text-muted-foreground">{fmtSize(size(k))}</span>
            </figcaption>
          </figure>
        ))}
      </div>

      <div className="mt-5 grid gap-4 border-t pt-5 md:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">Processing time (measured)</p>
          <div className="num flex flex-wrap gap-x-5 gap-y-1 text-sm">
            <span>EDSR-lite <b>{fmtMs(r.timing_ms.edsr)}</b></span>
            <span>Bicubic <b>{fmtMs(r.timing_ms.bicubic)}</b></span>
            <span>Total <b>{fmtMs(r.timing_ms.total)}</b></span>
          </div>
        </div>
        <div>
          <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">Quality</p>
          {r.metrics ? (
            <div className="num flex flex-wrap gap-x-5 gap-y-1 text-sm">
              <span>EDSR <b>{r.metrics.edsr.psnr_db.toFixed(2)} dB</b> / <b>{r.metrics.edsr.ssim.toFixed(4)}</b></span>
              <span className="text-muted-foreground">Bicubic {r.metrics.bicubic.psnr_db.toFixed(2)} dB / {r.metrics.bicubic.ssim.toFixed(4)}</span>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No ground-truth reference: PSNR/SSIM unavailable. Use Benchmark mode to score.</p>
          )}
        </div>
      </div>

      <div className="mt-5 grid gap-4 border-t pt-5 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-sm font-medium">EDSR-lite ×4</p>
          <Downloads png={r.images.edsr} name={`${name}_edsr_x4`} />
        </div>
        <div>
          <p className="mb-2 text-sm font-medium">Bicubic ×4</p>
          <Downloads png={r.images.bicubic} name={`${name}_bicubic_x4`} />
        </div>
      </div>

      {(r.notes.length > 0 || r.source.converted_to_grayscale) && (
        <div className="mt-5 space-y-1.5 border-t pt-4">
          {r.source.converted_to_grayscale && (
            <p className="text-xs text-muted-foreground"><Badge variant="outline" className="mr-2">Note</Badge>
              {r.source.mode} image converted to grayscale luminance; the model is single-channel.</p>
          )}
          {r.notes.map((n) => (
            <p key={n} className="text-xs text-muted-foreground"><Badge variant="outline" className="mr-2">Note</Badge>{n}</p>
          ))}
        </div>
      )}
    </Section>
  )
}
