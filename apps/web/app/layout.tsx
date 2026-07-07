import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "VARLens AI",
  description: "Educational foul and sanction analysis for short soccer clips.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
