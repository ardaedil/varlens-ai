export const dynamic = "force-dynamic"

export async function GET() {
  const apiBase = process.env.API_BASE_URL ?? "http://localhost:8000"

  try {
    const response = await fetch(`${apiBase}/api/v1/health`, { cache: "no-store" })
    const payload = await response.json()
    return Response.json(payload, { status: response.status })
  } catch {
    return Response.json(
      {
        service: "varlens-api",
        status: "degraded",
        model_available: false,
      },
      { status: 503 },
    )
  }
}
