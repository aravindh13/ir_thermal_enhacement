import { useState } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { KeyValue, MetricCard, PageHeader, Section } from "@/components/common"
import type { ModelCard } from "@/lib/api"
import { fmtMB, fmtParams } from "@/lib/format"

function Block({ title, detail, tone = "default" }: { title: string; detail?: string; tone?: "default" | "accent" | "muted" }) {
  const c = tone === "accent" ? "border-primary/30 bg-primary/5" : tone === "muted" ? "border-dashed bg-muted/50" : "bg-white"
  return (
    <div className={`rounded-lg border px-3 py-2 text-center ${c}`}>
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="num text-xs text-muted-foreground">{detail}</p>}
    </div>
  )
}
const Arrow = () => <div className="text-center text-muted-foreground" aria-hidden>↓</div>

function Architecture({ a }: { a: ModelCard["architecture"] }) {
  return (
    <div className="grid gap-6 md:grid-cols-[1fr_200px]" role="img"
      aria-label="EDSR-lite: head convolution, 16 residual blocks, body convolution with feature skip, two PixelShuffle x2 stages, output convolution, plus a bicubic x4 global skip.">
      <div className="space-y-1.5">
        <Block title="LR input" detail="1 channel, H×W" tone="muted" />
        <Arrow />
        <Block title="Head conv 3×3" detail={`1 → ${a.features} channels`} />
        <Arrow />
        <div className="rounded-xl border border-dashed p-2">
          <p className="mb-1.5 text-center text-xs text-muted-foreground">Body · feature skip around this block</p>
          <Block title={`${a.residual_blocks} × residual block`} detail={`${a.block} · ×${a.residual_scale}`} tone="accent" />
          <div className="mt-1.5"><Block title="Body conv 3×3" detail={`${a.features} → ${a.features}`} /></div>
        </div>
        <Arrow />
        <Block title="Upsampler" detail={String(a.upsampler)} tone="accent" />
        <Arrow />
        <Block title="Output conv 3×3" detail={`${a.features} → 1`} />
        <Arrow />
        <Block title="+ Bicubic ×4 of input" detail="global skip" tone="muted" />
        <Arrow />
        <Block title="SR output" detail="1 channel, 4H×4W" tone="muted" />
      </div>
      <div className="space-y-3 text-sm text-muted-foreground">
        <p><b className="text-foreground">Residual blocks</b> learn corrections on top of their input; scaling by {a.residual_scale} keeps training stable without batch norm.</p>
        <p><b className="text-foreground">PixelShuffle</b> turns 4× channels into 2× resolution. Two stages give ×4 with learned sub-pixel upsampling instead of a fixed interpolation kernel.</p>
        <p><b className="text-foreground">Global skip</b>: the network only predicts the detail that bicubic misses ({String(a.global_skip)}).</p>
      </div>
    </div>
  )
}

function Figures() {
  const [ok, setOk] = useState<Record<number, boolean>>({})
  const shown = [0, 1, 2, 3, 4].filter((i) => ok[i] !== false)
  if (shown.length === 0) return null
  return (
    <Section className="mt-6" title="Reference comparisons from the final evaluation"
      description="figures/cmp_*.png produced by train_sr.evaluate(): bicubic | EDSR-lite | ground truth (LLVIP test images).">
      <div className="space-y-4">
        {shown.map((i) => (
          <img key={i} src={`/api/figures/${i}`} alt={`Evaluation comparison ${i}: bicubic, EDSR-lite, ground truth`}
            loading="lazy" className="w-full rounded-lg border bg-zinc-100"
            onLoad={() => setOk((o) => ({ ...o, [i]: true }))} onError={() => setOk((o) => ({ ...o, [i]: false }))} />
        ))}
      </div>
    </Section>
  )
}

export function ModelInfo({ card }: { card: ModelCard | null }) {
  if (!card) return <PageHeader title="Model information" description="Model card unavailable: the backend is not reachable." />
  const t = card.training, d = card.dataset
  return (
    <>
      <PageHeader title="Model information" description="EDSR-lite (Enhanced Deep Residual Super-Resolution, reduced size), trained from scratch for ×4 thermal super-resolution." />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Parameters" value={fmtParams(card.parameters)} sub={card.parameters.toLocaleString()} />
        <MetricCard label="Weights" value={fmtMB(card.weights_bytes)} sub="edsr_x4.pt" />
        <MetricCard label="Residual blocks" value={card.architecture.residual_blocks} sub={`${card.architecture.features} feature channels`} />
        <MetricCard label="Scale" value={`×${card.scale}`} sub="2 × PixelShuffle ×2" />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Section title="Architecture"><Architecture a={card.architecture} /></Section>
        <div className="space-y-6">
          <Section title="Training configuration">
            <KeyValue rows={[
              ["Loss", t.loss], ["Optimizer", `${t.optimizer}, lr ${t.learning_rate}`], ["Batch size", t.batch_size],
              ["Patches", `${t.hr_patch}² HR / ${t.lr_patch}² LR`], ["Augmentation", t.augmentation],
              ["Duration", `${t.epochs} epochs · ${t.minutes} min`], ["Hardware", t.gpu], ["Initialisation", t.init],
            ]} />
          </Section>
          <Section title="Dataset" description={<a className="underline underline-offset-2" href={String(d.url)} target="_blank" rel="noreferrer">{String(d.name)}</a>}>
            <KeyValue rows={[
              ["Train / test images", `${Number(d.train_images).toLocaleString()} / ${Number(d.test_images).toLocaleString()}`],
              ["Used for training", Number(d.train_subset).toLocaleString()], ["Used for evaluation", d.eval_subset],
              ["Resolution", d.resolution], ["Channels", d.channels], ["LR generation", d.degradation],
            ]} />
          </Section>
        </div>
      </div>

      <Alert className="mt-6">
        <AlertTitle>Current limitations</AlertTitle>
        <AlertDescription>
          <ul className="mt-1 list-disc space-y-1 pl-4">{card.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        </AlertDescription>
      </Alert>

      <Figures />
    </>
  )
}
