// Before/after slider on top of the shared zoom/pan view. Left of the handle shows `left`, right shows `right`.
import { useRef, useState } from "react"
import { Layer, Viewport, type ViewCtl } from "./ZoomPan"

type Side = { src: string; label: string; pixelated?: boolean }

export function CompareSlider({ ctl, aspect, left, right }: { ctl: ViewCtl; aspect: number; left: Side; right: Side }) {
  const [pos, setPos] = useState(50)
  const box = useRef<HTMLDivElement>(null)

  const setFromX = (clientX: number) => {
    const r = box.current?.getBoundingClientRect()
    if (r) setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }

  return (
    <div ref={box}>
      <Viewport ctl={ctl} aspect={aspect}>
        <Layer src={left.src} view={ctl.view} alt={left.label} pixelated={left.pixelated} />
        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
          <Layer src={right.src} view={ctl.view} alt={right.label} pixelated={right.pixelated} />
        </div>
        <span className="pointer-events-none absolute top-2 left-2 rounded-md bg-white/90 px-2 py-0.5 text-xs font-medium shadow-sm">
          {left.label}
        </span>
        <span className="pointer-events-none absolute top-2 right-2 rounded-md bg-white/90 px-2 py-0.5 text-xs font-medium shadow-sm">
          {right.label}
        </span>
        <div className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_0_1px_rgba(0,0,0,.15)]" style={{ left: `${pos}%` }} />
        <div
          data-no-pan
          role="slider"
          tabIndex={0}
          aria-label="Comparison position"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pos)}
          aria-valuetext={`${Math.round(pos)}% ${left.label}`}
          className="absolute top-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full border bg-white text-xs text-zinc-600 shadow-md outline-none focus-visible:ring-3 focus-visible:ring-ring/60"
          style={{ left: `${pos}%` }}
          onPointerDown={(e) => {
            e.stopPropagation()
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerMove={(e) => e.currentTarget.hasPointerCapture(e.pointerId) && setFromX(e.clientX)}
          onKeyDown={(e) => {
            const d = e.key === "ArrowLeft" ? -2 : e.key === "ArrowRight" ? 2 : e.key === "Home" ? -100 : e.key === "End" ? 100 : 0
            if (d) {
              e.preventDefault()
              e.stopPropagation() // don't also pan the viewport
              setPos((p) => Math.min(100, Math.max(0, p + d)))
            }
          }}
        >
          ⇆
        </div>
      </Viewport>
    </div>
  )
}
