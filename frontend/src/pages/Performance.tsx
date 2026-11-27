import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Page } from "@/components/AppShell"
import { EmptyState, KeyValue, MetricBars, MetricCard, PageHeader, Section } from "@/components/common"
import type { ModelCard, Result } from "@/lib/api"
import { fmtMB, fmtMs, fmtParams, fmtSize, fmtTime, METHOD } from "@/lib/format"
import type { Run } from "@/lib/history"

const STAGES: Record<string, string> = { bicubic: "Bicubic", edsr: "EDSR-lite", metrics: "PSNR/SSIM", encode: "PNG encode" }

export function Performance({ card, result: r, runs, onClear, go }: {
  card: ModelCard | null; result: Result | null; runs: Run[]; onClear: () => void; go: (p: Page) => void
}) {
  const points = runs.map((x) => ({ mp: +((x.input.width * x.input.height) / 1e6).toFixed(3), s: +(x.edsr_ms / 1000).toFixed(2), name: x.filename }))
  const rep = card?.reported_results

  return (
    <>
      <PageHeader title="Performance analysis" description="Every number on this page is measured on this machine, computed against a ground-truth image, or quoted from REPORT.md (labelled)." />

      <h2 className="mb-3 text-sm font-semibold">This run</h2>
      {!r ? (
        <EmptyState title="No run in this session">
          <Button variant="link" onClick={() => go("workspace")}>Process an image</Button>
        </EmptyState>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <MetricCard label="EDSR-lite time" value={fmtMs(r.timing_ms.edsr)} sub={`on ${r.device.toUpperCase()}`} />
            <MetricCard label="Total request" value={fmtMs(r.timing_ms.total)} sub="server-side, excl. network" />
            <MetricCard label="Resolution" value={`${fmtSize(r.input)} → ${fmtSize(r.output)}`} sub={`×${r.scale} per side · ×${r.scale ** 2} pixels`} />
            <MetricCard label="Model size" value={card ? fmtMB(card.weights_bytes) : "—"} sub={card ? `${fmtParams(card.parameters)} parameters` : undefined} />
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Section title="Processing time by stage" description={`${r.filename} · ${r.mode}`}>
              <MetricBars unit="ms" digits={0} data={Object.entries(STAGES).filter(([k]) => k in r.timing_ms).map(([k, name]) => ({
                name, value: r.timing_ms[k], color: k === "edsr" ? METHOD.edsr.color : k === "bicubic" ? METHOD.bicubic.color : "var(--c-input)",
              }))} />
            </Section>

            <Section title="Quality vs ground truth" description={r.metrics ? `Reference: ${r.metrics.reference}` : undefined}>
              {r.metrics ? (
                <div className="space-y-5">
                  <div>
                    <p className="mb-1 text-xs font-medium text-muted-foreground">PSNR (dB)</p>
                    <MetricBars unit="dB" data={[
                      { name: METHOD.bicubic.label, value: r.metrics.bicubic.psnr_db, color: METHOD.bicubic.color },
                      { name: METHOD.edsr.label, value: r.metrics.edsr.psnr_db, color: METHOD.edsr.color },
                    ]} />
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium text-muted-foreground">SSIM</p>
                    <MetricBars digits={4} domain={[0, 1]} data={[
                      { name: METHOD.bicubic.label, value: r.metrics.bicubic.ssim, color: METHOD.bicubic.color },
                      { name: METHOD.edsr.label, value: r.metrics.edsr.ssim, color: METHOD.edsr.color },
                    ]} />
                  </div>
                  <p className="num text-sm">
                    Gain over bicubic: <b>{(r.metrics.edsr.psnr_db - r.metrics.bicubic.psnr_db >= 0 ? "+" : "")}{(r.metrics.edsr.psnr_db - r.metrics.bicubic.psnr_db).toFixed(2)} dB</b> PSNR,{" "}
                    <b>{(r.metrics.edsr.ssim - r.metrics.bicubic.ssim >= 0 ? "+" : "")}{(r.metrics.edsr.ssim - r.metrics.bicubic.ssim).toFixed(4)}</b> SSIM
                  </p>
                </div>
              ) : (
                <Alert>
                  <AlertTitle>No ground-truth reference</AlertTitle>
                  <AlertDescription>
                    This was an Enhance run, so there is no high-resolution reference to compare against and PSNR/SSIM cannot be computed honestly.
                    Use Benchmark mode with a high-resolution image to score the model.
                  </AlertDescription>
                </Alert>
              )}
            </Section>
          </div>
        </>
      )}

      <h2 className="mt-8 mb-3 text-sm font-semibold">This browser</h2>
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="EDSR-lite time vs input size" description="Each point is a real run from this browser.">
          {points.length === 0 ? <EmptyState title="No runs recorded" /> : (
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart margin={{ left: 0, right: 16, top: 8, bottom: 16 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis type="number" dataKey="mp" name="Input" unit=" MP" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)"
                  label={{ value: "Input megapixels", position: "insideBottom", offset: -8, fontSize: 11 }} />
                <YAxis type="number" dataKey="s" name="EDSR time" unit=" s" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" width={48} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter data={points} fill="var(--c-edsr)" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </Section>

        <Section title="Reported results" description={rep?.source}>
          {rep ? (
            <Table>
              <TableHeader><TableRow><TableHead>Method</TableHead><TableHead className="text-right">PSNR</TableHead><TableHead className="text-right">SSIM</TableHead></TableRow></TableHeader>
              <TableBody>
                {(["bicubic", "edsr"] as const).map((k) => (
                  <TableRow key={k}>
                    <TableCell className="font-medium">{METHOD[k].label}</TableCell>
                    <TableCell className="num text-right">{rep[k].psnr_db.toFixed(2)} dB</TableCell>
                    <TableCell className="num text-right">{rep[k].ssim.toFixed(4)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : <p className="text-sm text-muted-foreground">Model card unavailable.</p>}
          <KeyValue className="mt-3" rows={[["Evaluation set", "100 evenly spaced LLVIP test images"], ["Degradation", "bicubic ×4 (synthetic)"]]} />
        </Section>
      </div>

      <Section
        className="mt-6"
        title="Run history"
        description={`${runs.length} run${runs.length === 1 ? "" : "s"} stored in this browser (max 50).`}
        action={runs.length > 0 && <Button variant="ghost" size="sm" onClick={onClear}>Clear</Button>}
      >
        {runs.length === 0 ? <EmptyState title="No runs recorded" /> : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead><TableHead>File</TableHead><TableHead>Mode</TableHead><TableHead>Bits</TableHead>
                <TableHead>Input → output</TableHead><TableHead className="text-right">EDSR</TableHead><TableHead className="text-right">Total</TableHead>
                <TableHead className="text-right">PSNR bic / EDSR</TableHead><TableHead className="text-right">SSIM bic / EDSR</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.map((x) => (
                <TableRow key={x.id}>
                  <TableCell className="text-muted-foreground">{fmtTime(x.time)}</TableCell>
                  <TableCell className="max-w-40 truncate font-medium">{x.filename}</TableCell>
                  <TableCell><Badge variant="outline">{x.mode}</Badge></TableCell>
                  <TableCell className="num">{x.bit_depth}</TableCell>
                  <TableCell className="num">{fmtSize(x.input)} → {fmtSize(x.output)}</TableCell>
                  <TableCell className="num text-right">{fmtMs(x.edsr_ms)}</TableCell>
                  <TableCell className="num text-right">{fmtMs(x.total_ms)}</TableCell>
                  <TableCell className="num text-right">{x.metrics ? `${x.metrics.bicubic.psnr_db.toFixed(2)} / ${x.metrics.edsr.psnr_db.toFixed(2)}` : "—"}</TableCell>
                  <TableCell className="num text-right">{x.metrics ? `${x.metrics.bicubic.ssim.toFixed(4)} / ${x.metrics.edsr.ssim.toFixed(4)}` : "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Section>
    </>
  )
}
