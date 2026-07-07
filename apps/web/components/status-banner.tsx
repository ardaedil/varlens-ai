import type { ReactNode } from "react"

interface StatusBannerProps {
  title: string
  tone: "neutral" | "warning" | "error"
  children: ReactNode
}

export function StatusBanner({ title, tone, children }: StatusBannerProps) {
  return (
    <div className={`banner ${tone === "neutral" ? "" : tone}`} role={tone === "error" ? "alert" : "status"}>
      <strong>{title}</strong>
      <div>{children}</div>
    </div>
  )
}
