import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { Page } from "@/components/AppShell"
import { CompareSlider } from "@/components/CompareSlider"
import { EmptyState, PageHeader, Section } from "@/components/common"
import { Layer, useView, Viewport, ZoomControls } from "@/components/ZoomPan"
import type { Result } from "@/lib/api"
import { fmtSize, METHOD, type MethodKey } from "@/lib/format"

function Picker({ keys, value, onChange, name }: {
  keys: MethodKey[]; value: MethodKey; onChange: (k: MethodKey) => void; name: string
}) {
  return (
    <ToggleGroup variant="outline" size="sm" spacing={0} value={[value]} aria-label={name}
      onValueChange={(v) => v[0] && onChange(v[0] as MethodKey)}>
      {keys.map((k) => <ToggleGroupItem key={k} value={k}>{k === "input" ? "Original" : METHOD[k].label}</ToggleGroupItem>)}
    </ToggleGroup>
  )
}

export function Compare({ result: r, go }: { result: Result | null; go: (p: Page) => void }) {
  const ctl = useView()
  const [left, setLeft] = useState<MethodKey>("bicubic")
  const [right, setRight] = useState<MethodKey>("edsr")

  if (!r)
    return (
      <>
        <PageHeader title="Comparison" />
        <EmptyState title="Nothing to compare yet">
          <Button variant="link" onClick={() => go("workspace")}>Process an image first</Button>
        </EmptyState>
      </>
    )

  const keys: MethodKey[] = r.images.ground_truth ? ["input", "bicubic", "edsr", "ground_truth"] : ["input", "bicubic", "edsr"]
  const aspect = r.output.width / r.output.height
  const label = (k: MethodKey) =>
    `${k === "input" && r.mode === "benchmark" ? "Synthetic LR" : METHOD[k].label} · ${fmtSize(k === "input" ? r.input : r.output)}`
  const side = (k: MethodKey) => ({ src: r.images[k]!.png8, label: label(k), pixelated: k === "input" })

  return (
    <>
      <PageHeader
        title="Comparison"
        description={`${r.filename} · ${r.mode} · ${fmtSize(r.input)} → ${fmtSize(r.output)}. Zoom and pan are synchronised across every view.`}
      />
      <Tabs defaultValue="slider">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <TabsList>
            <TabsTrigger value="slider">Slider</TabsTrigger>
            <TabsTrigger value="side">Side by side</TabsTrigger>
          </TabsList>
          <ZoomControls ctl={ctl} />
        </div>

        <TabsContent value="slider">
          <Section
            title="Before / after"
            description="Drag the handle (or focus it and use ← →). Scroll to zoom, drag to pan, double-click to fit."
          >
            <div className="mb-4 flex flex-wrap gap-x-6 gap-y-3">
              <div className="space-y-1.5"><p className="text-xs font-medium text-muted-foreground">Left</p><Picker keys={keys} name="Left image" value={left} onChange={setLeft} /></div>
              <div className="space-y-1.5"><p className="text-xs font-medium text-muted-foreground">Right</p><Picker keys={keys} name="Right image" value={right} onChange={setRight} /></div>
            </div>
            <div className="mx-auto max-w-5xl">
              <CompareSlider ctl={ctl} aspect={aspect} left={side(left)} right={side(right)} />
            </div>
          </Section>
        </TabsContent>

        <TabsContent value="side">
          <div className={`grid gap-4 md:grid-cols-2 ${keys.length === 4 ? "2xl:grid-cols-4" : "xl:grid-cols-3"}`}>
            {keys.map((k) => (
              <figure key={k} className="space-y-2">
                <Viewport ctl={ctl} aspect={aspect} label={k === "input" && r.mode === "benchmark" ? "Synthetic LR" : METHOD[k].label}>
                  <Layer src={r.images[k]!.png8} view={ctl.view} alt={METHOD[k].label} pixelated={k === "input"} />
                </Viewport>
                <figcaption className="flex justify-between text-xs">
                  <span className="flex items-center gap-1.5 font-medium">
                    <span className="size-2 rounded-sm" style={{ background: METHOD[k].color }} aria-hidden />
                    {k === "input" && r.mode === "benchmark" ? "Synthetic LR input" : METHOD[k].label}
                  </span>
                  <span className="num text-muted-foreground">{fmtSize(k === "input" ? r.input : r.output)}</span>
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            The original is shown with nearest-neighbour scaling at the output size so pixels line up with the ×4 results.
          </p>
        </TabsContent>
      </Tabs>
    </>
  )
}
