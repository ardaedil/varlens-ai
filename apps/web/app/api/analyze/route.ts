export const dynamic = "force-dynamic"

export async function POST(request: Request) {
  const apiBase = process.env.API_BASE_URL ?? "http://localhost:8000"
  const formData = await request.formData()

  try {
    const response = await fetch(`${apiBase}/api/v1/analyze`, {
      method: "POST",
      body: formData,
    })
    const payload = await response.json()
    return Response.json(payload, { status: response.status })
  } catch {
    return Response.json(
      {
        spec_version: "1.0.0",
        request_id: crypto.randomUUID(),
        status: "error",
        error: {
          code: "model_unavailable",
          message: "The VARLens API is not reachable.",
        },
      },
      { status: 503 },
    )
  }
}
