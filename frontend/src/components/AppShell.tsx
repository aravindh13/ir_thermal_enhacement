import type { ReactNode } from "react"
import { BarChart3, Cpu, GitCompare, LayoutDashboard, Sparkles } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { Health } from "@/lib/api"
import { cn } from "@/lib/utils"

export type Page = "dashboard" | "workspace" | "compare" | "performance" | "model"

const NAV: { id: Page; label: string; icon: typeof Cpu }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "workspace", label: "Enhance", icon: Sparkles },
  { id: "compare", label: "Compare", icon: GitCompare },
  { id: "performance", label: "Performance", icon: BarChart3 },
  { id: "model", label: "Model", icon: Cpu },
]

function Status({ health }: { health: Health | null | "error" }) {
  if (health === null) return <Badge variant="outline">Connecting…</Badge>
  if (health === "error" || !health.model_loaded)
    return <Badge variant="destructive">Backend offline</Badge>
  return (
    <Badge variant="outline" className="gap-1.5">
      <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden />
      Model ready · {health.device.toUpperCase()}
    </Badge>
  )
}

export function AppShell({ page, onNavigate, health, children }: {
  page: Page; onNavigate: (p: Page) => void; health: Health | null | "error"; children: ReactNode
}) {
  return (
    <div className="min-h-svh lg:grid lg:grid-cols-[240px_1fr]">
      <div className="sticky top-0 z-10 bg-white lg:static lg:border-r">{/* full-height column background */}
      <aside className="border-b bg-white lg:sticky lg:top-0 lg:h-svh lg:border-b-0">
        <div className="flex items-center justify-between gap-3 px-5 py-4 lg:block">
          <div>
            <p className="text-sm font-semibold tracking-tight">IR Thermal SR</p>
            <p className="text-xs text-muted-foreground">EDSR-lite ×4 research demo</p>
          </div>
          <div className="lg:mt-3"><Status health={health} /></div>
        </div>
        <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-3 pb-3 [scrollbar-width:none] lg:flex-col lg:pb-0">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              aria-current={page === id ? "page" : undefined}
              className={cn(
                "flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50",
                page === id && "bg-primary/8 text-primary hover:bg-primary/10 hover:text-primary",
              )}
            >
              <Icon className="size-4" aria-hidden />
              {label}
            </button>
          ))}
        </nav>
      </aside>
      </div>
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-8 sm:py-8">
        {health === "error" && (
          <div role="alert" className="mb-6 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm">
            The backend is not reachable. Start it from the repo root with{" "}
            <code className="rounded bg-white px-1.5 py-0.5 font-mono text-xs">uvicorn backend.app:app</code>, then reload.
          </div>
        )}
        {children}
      </main>
    </div>
  )
}
