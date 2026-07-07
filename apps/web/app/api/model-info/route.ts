export const dynamic = "force-dynamic"

export async function GET() {
  const apiBase = process.env.API_BASE_URL ?? "http://localhost:8000"

  try {
    const response = await fetch(`${apiBase}/api/v1/model-info`, { cache: "no-store" })
    const payload = await response.json()
    return Response.json(payload, { status: response.status })
  } catch {
    return Response.json(
      {
        spec_version: "1.0.0",
        model_version: "unavailable",
        supported_scopes: ["foul_review_context"],
        unsupported_scopes: ["offside", "handball", "penalty_no_penalty"],
        limitations: ["The backend is not reachable."],
      },
      { status: 503 },
    )
  }
}
