"use client"

import { useEffect, useMemo, useState } from "react"
import type { AnalysisScope, AnalyzeEnvelope } from "@varlens/contracts"
import { isErrorResponse } from "@varlens/contracts"
import { ResultCard } from "@/components/result-card"
import { StatusBanner } from "@/components/status-banner"

const scopes: Array<{ value: AnalysisScope; label: string }> = [
  { value: "foul_review_context", label: "Foul" },
  { value: "offside", label: "Offside" },
  { value: "handball", label: "Handball" },
]

export function AnalyzeWorkbench() {
  const [file, setFile] = useState<File | null>(null)
  const [scope, setScope] = useState<AnalysisScope>("foul_review_context")
  const [duration, setDuration] = useState("8")
  const [result, setResult] = useState<AnalyzeEnvelope | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [health, setHealth] = useState<"checking" | "ok" | "degraded">("checking")

  useEffect(() => {
    let cancelled = false
    fetch("/api/health")
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then((payload) => {
        if (!cancelled) setHealth(payload.status === "ok" ? "ok" : "degraded")
      })
      .catch(() => {
        if (!cancelled) setHealth("degraded")
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const selectedFileLabel = useMemo(() => {
    if (!file) return "No clip selected"
    const mb = file.size / (1024 * 1024)
    return `${file.name} - ${mb.toFixed(1)} MB`
  }, [file])

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)
    formData.append("scope", scope)
    if (duration.trim()) {
      formData.append("clip_duration_seconds", duration)
    }

    setIsSubmitting(true)
    setResult(null)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })
      const payload = (await response.json()) as AnalyzeEnvelope
      setResult(payload)
    } catch {
      setResult({
        spec_version: "1.0.0",
        request_id: crypto.randomUUID(),
        status: "error",
        error: {
          code: "model_unavailable",
          message: "The analysis service is not reachable.",
        },
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <strong>VARLens AI</strong>
          <span>Short-clip foul and sanction analysis</span>
        </div>
        <div className="system-pill" aria-live="polite">
          <span className="status-dot" aria-hidden="true" />
          API {health}
        </div>
      </header>

      <div className="main-grid">
        <section className="panel control-panel" aria-labelledby="analysis-controls-title">
          <div className="section-title">
            <div>
              <h1 id="analysis-controls-title">Analysis</h1>
              <p>One uploaded clip, one v1 scope, no official decision claim.</p>
            </div>
          </div>

          <form className="field-stack" onSubmit={handleSubmit}>
            <div className="field">
              <label htmlFor="clip">Clip</label>
              <div className="file-drop">
                <input
                  id="clip"
                  name="clip"
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime,video/x-matroska,.mov,.mkv"
                  onChange={(event) => setFile(event.currentTarget.files?.[0] ?? null)}
                />
              </div>
              <p className="helper">{selectedFileLabel}</p>
            </div>

            {previewUrl ? (
              <div className="preview">
                <video controls src={previewUrl} />
              </div>
            ) : null}

            <fieldset className="field">
              <legend className="legend">Scope</legend>
              <div className="segmented">
                {scopes.map((item) => (
                  <label className="segment" key={item.value}>
                    <input
                      type="radio"
                      name="scope"
                      value={item.value}
                      checked={scope === item.value}
                      onChange={() => setScope(item.value)}
                    />
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="field">
              <label htmlFor="duration">Duration estimate</label>
              <div className="duration-row">
                <input
                  id="duration"
                  inputMode="decimal"
                  value={duration}
                  onChange={(event) => setDuration(event.currentTarget.value)}
                />
                <span className="helper">seconds</span>
              </div>
            </div>

            <button className="primary-button" disabled={!file || isSubmitting} type="submit">
              {isSubmitting ? "Analyzing..." : "Analyze clip"}
            </button>
          </form>
        </section>

        <section className="panel results-panel" aria-live="polite" aria-labelledby="results-title">
          <div className="section-title">
            <div>
              <h2 id="results-title">Result</h2>
              <p>Predictions, alternatives, focus points, and limitations.</p>
            </div>
          </div>

          {result ? (
            isErrorResponse(result) ? (
              <StatusBanner tone="error" title={result.error.code}>
                {result.error.message}
              </StatusBanner>
            ) : (
              <ResultCard result={result} />
            )
          ) : (
            <div className="empty-state">
              <h2>Foul review context, with uncertainty left visible.</h2>
              <p>
                VARLens v1 reports sanction likelihood, action type, viewer focus points,
                limitations, and model provenance for short clips.
              </p>
              <StatusBanner tone="warning" title="v1 boundary">
                Offside, handball, and penalty/no-penalty scopes return unsupported results.
              </StatusBanner>
            </div>
          )}
        </section>
      </div>
    </main>
  )
}
