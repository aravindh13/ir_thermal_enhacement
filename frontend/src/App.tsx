import { useEffect, useState } from "react"
import { AppShell, type Page } from "@/components/AppShell"
import { getHealth, getModel, type Health, type ModelCard, type Result } from "@/lib/api"
import { addRun, clearRuns, loadRuns, type Run } from "@/lib/history"
import { Compare } from "@/pages/Compare"
import { Dashboard } from "@/pages/Dashboard"
import { ModelInfo } from "@/pages/ModelInfo"
import { Performance } from "@/pages/Performance"
import { Workspace } from "@/pages/Workspace"

const PAGES: Page[] = ["dashboard", "workspace", "compare", "performance", "model"]
const fromHash = (): Page => {
  const h = location.hash.slice(1) as Page
  return PAGES.includes(h) ? h : "dashboard"
}

export default function App() {
  // ponytail: hash-based page state; add react-router if real deep links are needed
  const [page, setPage] = useState<Page>(fromHash)
  const [health, setHealth] = useState<Health | null | "error">(null)
  const [card, setCard] = useState<ModelCard | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [runs, setRuns] = useState<Run[]>(loadRuns)

  useEffect(() => {
    getHealth().then(setHealth, () => setHealth("error"))
    getModel().then(setCard, () => setCard(null))
    const onHash = () => setPage(fromHash())
    addEventListener("hashchange", onHash)
    return () => removeEventListener("hashchange", onHash)
  }, [])

  const go = (p: Page) => {
    location.hash = p
    setPage(p)
    scrollTo({ top: 0 })
  }
  const onResult = (r: Result) => {
    setResult(r)
    setRuns(addRun(r))
  }

  return (
    <AppShell page={page} onNavigate={go} health={health}>
      {page === "dashboard" && <Dashboard card={card} health={health} runs={runs} go={go} />}
      {page === "workspace" && <Workspace card={card} result={result} onResult={onResult} go={go} />}
      {page === "compare" && <Compare result={result} go={go} />}
      {page === "performance" && (
        <Performance card={card} result={result} runs={runs} onClear={() => setRuns(clearRuns())} go={go} />
      )}
      {page === "model" && <ModelInfo card={card} />}
    </AppShell>
  )
}
