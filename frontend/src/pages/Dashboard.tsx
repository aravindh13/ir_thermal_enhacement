import { ArrowRight } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Page } from "@/components/AppShell"
import { EmptyState, MetricBars, MetricCard, PageHeader, Section } from "@/components/common"
import type { Health, ModelCard } from "@/lib/api"
import { fmtMs, fmtParams, fmtSize, fmtTime, METHOD } from "@/lib/format"
import type { Run } from "@/lib/history"

export function Dashboard({ card, health, runs, go }: {
  card: ModelCard | null; health: Health | null | "error"; runs: Run[]; go: (p: Page) => void
}) {
  const ready = health !== null && health !== "error" && health.model_loaded
  const rep = card?.reported_results
  const t = card?.training, d = card?.dataset

  return (
    <>
      <PageHeader
        title="IR Thermal Image Super-Resolution"
        description="A ×4 convolutional super-resolution network for grayscale infrared images, trained on LLVIP and compared against bicubic interpolation."
        action={<Button onClick={() => go("workspace")}>Enhance an image <ArrowRight data-icon="inline-end" /></Button>}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Model" value={card ? `${card.name} ×${card.scale}` : "—"} sub="Residual CNN, PixelShuffle upsampling" />
        <MetricCard label="Parameters" value={card ? fmtParams(card.parameters) : "—"} sub={card ? `${card.parameters.toLocaleString()} weights` : undefined} />
        <MetricCard label="Training" value={t ? `${t.epochs} epochs` : "—"} sub={t ? `${t.minutes} min · ${t.gpu}` : undefined} />
        <MetricCard label="Dataset" value="LLVIP infrared" sub={d ? `${Number(d.train_subset).toLocaleString()} training images` : undefined} />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Section
          className="lg:col-span-2"
          title="Reported results"
          description={rep ? `From ${rep.source}.` : "Loading…"}
        >
          {rep && (
            <div className="grid gap-6 sm:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-medium text-muted-foreground">PSNR (dB, higher is better)</p>
                <MetricBars unit="dB" domain={[30, 42]} data={[
                  { name: METHOD.bicubic.label, value: rep.bicubic.psnr_db, color: METHOD.bicubic.color },
                  { name: METHOD.edsr.label, value: rep.edsr.psnr_db, color: METHOD.edsr.color },
                ]} />
              </div>
              <div>
                <p className="mb-1 text-xs font-medium text-muted-foreground">SSIM (higher is better)</p>
                <MetricBars digits={4} domain={[0.9, 1]} data={[
                  { name: METHOD.bicubic.label, value: rep.bicubic.ssim, color: METHOD.bicubic.color },
                  { name: METHOD.edsr.label, value: rep.edsr.ssim, color: METHOD.edsr.color },
                ]} />
              </div>
              <p className="text-xs text-muted-foreground sm:col-span-2">
                Axes start at 30 dB and 0.90 to make the gap readable. EDSR-lite gains{" "}
                <span className="num font-medium text-foreground">+{(rep.edsr.psnr_db - rep.bicubic.psnr_db).toFixed(2)} dB</span> PSNR over bicubic.
              </p>
            </div>
          )}
        </Section>

        <Section title="Available methods" description="Only implemented scales are offered.">
          <ul className="space-y-3">
            {[["bicubic", "Interpolation baseline (PIL bicubic)"], ["edsr", "Trained EDSR-lite network"]].map(([k, desc]) => (
              <li key={k} className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5">
                  <span className="mt-1.5 size-2.5 shrink-0 rounded-sm" style={{ background: METHOD[k as "bicubic"].color }} aria-hidden />
                  <div>
                    <p className="text-sm font-medium">{METHOD[k as "bicubic"].label}</p>
                    <p className="text-xs text-muted-foreground">{desc}</p>
                  </div>
                </div>
                <Badge variant={ready ? "secondary" : "outline"}>{ready ? "Available" : "Offline"}</Badge>
              </li>
            ))}
          </ul>
          <p className="mt-4 border-t pt-3 text-xs text-muted-foreground">Scale ×4 only · grayscale input · CPU or GPU inference</p>
        </Section>
      </div>

      <Section className="mt-6" title="Recent activity" description="Runs from this browser (stored locally, never uploaded).">
        {runs.length === 0 ? (
          <EmptyState title="No runs yet">
            <Button variant="link" onClick={() => go("workspace")}>Enhance your first image</Button>
          </EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead><TableHead>File</TableHead><TableHead>Mode</TableHead>
                <TableHead>Input → output</TableHead><TableHead className="text-right">EDSR time</TableHead>
                <TableHead className="text-right">PSNR (EDSR)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.slice(0, 10).map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="text-muted-foreground">{fmtTime(r.time)}</TableCell>
                  <TableCell className="max-w-48 truncate font-medium">{r.filename}</TableCell>
                  <TableCell><Badge variant="outline">{r.mode}</Badge></TableCell>
                  <TableCell className="num">{fmtSize(r.input)} → {fmtSize(r.output)}</TableCell>
                  <TableCell className="num text-right">{fmtMs(r.edsr_ms)}</TableCell>
                  <TableCell className="num text-right">{r.metrics ? `${r.metrics.edsr.psnr_db.toFixed(2)} dB` : "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Section>
    </>
  )
}
