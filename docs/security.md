# Security

VARLens v1 follows OWASP ASVS and OWASP API Security Top 10 as the secure-development baseline.

## Upload Safety

- Accept only configured video media types.
- Enforce maximum size during streaming upload writes.
- Enforce maximum duration from client-provided metadata in v1.
- Store uploads in transient temp files only.
- Delete uploads after inference unless operator debug retention is enabled.

## API Safety

- Keep admin/operator endpoints out of v1 except read-only health/model-info.
- Return typed error envelopes instead of stack traces.
- Do not expose raw file paths or signed object URLs in responses.
- Add rate limiting before public deployment.
- Use least-privilege storage credentials if file storage is added later.

## Logging

Allowed:

- Request path, method, status, latency, request ID.
- Model version and aggregate outcome status.

Forbidden by default:

- Raw frames.
- Source video bytes.
- Public object URLs.
- Full copyrighted clip filenames when not needed for debugging.

## Deployment Gate

Before production deployment:

- Confirm no public third-party clip redistribution.
- Confirm rate limiting is configured.
- Confirm upload retention defaults to deletion.
- Confirm deployment credentials use OIDC or short-lived credentials where available.
