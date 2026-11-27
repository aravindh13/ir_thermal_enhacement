// Synchronised zoom/pan. View state is shared by every Viewport that receives the same `view`,
// expressed in fractions of the viewport so panels of different pixel sizes stay aligned.
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { cn } from "@/lib/utils"

export type View = { s: number; x: number; y: number } // scale; translate as fraction of viewport (<= 0)
export const MIN_ZOOM = 1
export const MAX_ZOOM = 8
const FIT: View = { s: 1, x: 0, y: 0 }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const bound = (v: View): View => ({ s: v.s, x: clamp(v.x, 1 - v.s, 0), y: clamp(v.y, 1 - v.s, 0) })

export function useView() {
  const [view, setView] = useState<View>(FIT)
  /** Zoom by `factor` keeping the point at viewport fraction (fx, fy) fixed. */
  const zoomAt = useCallback((factor: number, fx = 0.5, fy = 0.5) => {
    setView((v) => {
      const s = clamp(v.s * factor, MIN_ZOOM, MAX_ZOOM)
      const px = (fx - v.x) / v.s, py = (fy - v.y) / v.s
      return bound({ s, x: fx - px * s, y: fy - py * s })
    })
  }, [])
  const panBy = useCallback((dx: number, dy: number) => setView((v) => bound({ ...v, x: v.x + dx, y: v.y + dy })), [])
  const reset = useCallback(() => setView(FIT), [])
  return { view, zoomAt, panBy, reset }
}
export type ViewCtl = ReturnType<typeof useView>

type ViewportProps = {
  ctl: ViewCtl
  aspect: number // width / height of the full-resolution output
  label?: string
  children: ReactNode
  className?: string
}

/** Interactive frame: wheel zooms at cursor, drag pans, arrow keys pan, +/- zoom, 0 resets. */
export function Viewport({ ctl, aspect, label, children, className }: ViewportProps) {
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)
  const { zoomAt, panBy, reset } = ctl

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault() // needs a non-passive listener, which React's onWheel is not
      const r = el.getBoundingClientRect()
      zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [zoomAt])

  return (
    <div
      ref={ref}
      role="group"
      tabIndex={0}
      aria-label={`${label ?? "Image"} viewer. Scroll or press plus and minus to zoom, drag or use arrow keys to pan, 0 to reset.`}
      className={cn(
        "relative w-full cursor-grab touch-none overflow-hidden rounded-lg border bg-zinc-100 outline-none select-none active:cursor-grabbing focus-visible:ring-3 focus-visible:ring-ring/50",
        className,
      )}
      style={{ aspectRatio: aspect }}
      onPointerDown={(e) => {
        if ((e.target as HTMLElement).closest("[data-no-pan]")) return
        drag.current = { x: e.clientX, y: e.clientY }
        e.currentTarget.setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        if (!drag.current || !ref.current) return
        const r = ref.current.getBoundingClientRect()
        panBy((e.clientX - drag.current.x) / r.width, (e.clientY - drag.current.y) / r.height)
        drag.current = { x: e.clientX, y: e.clientY }
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
      onDoubleClick={reset}
      onKeyDown={(e) => {
        const step = 0.05
        const k: Record<string, () => void> = {
          ArrowLeft: () => panBy(step, 0),
          ArrowRight: () => panBy(-step, 0),
          ArrowUp: () => panBy(0, step),
          ArrowDown: () => panBy(0, -step),
          "+": () => zoomAt(1.25),
          "=": () => zoomAt(1.25),
          "-": () => zoomAt(1 / 1.25),
          "0": reset,
        }
        if (k[e.key]) {
          e.preventDefault()
          k[e.key]()
        }
      }}
    >
      {children}
      {label && (
        <span className="pointer-events-none absolute top-2 left-2 rounded-md bg-white/90 px-2 py-0.5 text-xs font-medium shadow-sm">
          {label}
        </span>
      )}
    </div>
  )
}

/** One image layer, stretched to the viewport and transformed by the shared view. */
export function Layer({ src, view, alt, pixelated }: { src: string; view: View; alt: string; pixelated?: boolean }) {
  return (
    <div
      className="absolute inset-0 origin-top-left will-change-transform"
      style={{ transform: `translate(${view.x * 100}%, ${view.y * 100}%) scale(${view.s})` }}
    >
      <img
        src={src}
        alt={alt}
        draggable={false}
        className={cn("h-full w-full", (pixelated || view.s > 1) && "pixelated")}
      />
    </div>
  )
}

export function ZoomControls({ ctl }: { ctl: ViewCtl }) {
  const b = "h-8 min-w-8 rounded-md border bg-white px-2 text-sm shadow-xs hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 outline-none"
  return (
    <div className="flex items-center gap-1.5">
      <button className={b} onClick={() => ctl.zoomAt(1 / 1.5)} aria-label="Zoom out">−</button>
      <span className="num w-14 text-center text-sm text-muted-foreground">{ctl.view.s.toFixed(1)}×</span>
      <button className={b} onClick={() => ctl.zoomAt(1.5)} aria-label="Zoom in">+</button>
      <button className={b} onClick={ctl.reset}>Fit</button>
    </div>
  )
}
