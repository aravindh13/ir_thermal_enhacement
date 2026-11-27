// Small presentational pieces shared by several pages.
import type { ReactNode } from "react"
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {action}
    </div>
  )
}

export function MetricCard({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <Card size="sm">
      <CardContent>
        <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{label}</p>
        <p className="num mt-1.5 text-xl font-semibold tracking-tight">{value}</p>
        {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
      </CardContent>
    </Card>
  )
}

export function Section({ title, description, children, className, action }: {
  title: string; description?: ReactNode; children: ReactNode; className?: string; action?: ReactNode
}) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
        {action && <CardAction>{action}</CardAction>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center">
      <p className="text-sm font-medium">{title}</p>
      {children && <div className="text-sm text-muted-foreground">{children}</div>}
    </div>
  )
}

export function KeyValue({ rows, className }: { rows: [string, ReactNode][]; className?: string }) {
  return (
    <dl className={cn("divide-y text-sm", className)}>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 py-2">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="num text-right font-medium">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export type BarDatum = { name: string; value: number; color: string }

/** Horizontal single-series bar chart with value labels; domain is caller-chosen so differences aren't exaggerated silently. */
export function MetricBars({ data, unit, domain, digits = 2, height }: {
  data: BarDatum[]; unit?: string; domain?: [number, number]; digits?: number; height?: number
}) {
  const fmt = (v: number) => `${v.toFixed(digits)}${unit ? ` ${unit}` : ""}`
  return (
    <ResponsiveContainer width="100%" height={height ?? 44 * data.length + 24}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 64, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--border)" />
        <XAxis type="number" domain={domain ?? [0, "auto"]} tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
        <YAxis type="category" dataKey="name" width={104} tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
        <Tooltip formatter={(v) => fmt(Number(v))} cursor={{ fill: "var(--muted)" }} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20} isAnimationActive={false}>
          {data.map((d) => <Cell key={d.name} fill={d.color} />)}
          <LabelList dataKey="value" position="right" formatter={(v) => fmt(Number(v))} className="num" fontSize={12} fill="var(--foreground)" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
