export const SPEC_VERSION = "1.0.0" as const
export const MODEL_VERSION = "videomae-mvfoul-v1-stub" as const

export const sanctionLabels = [
  "no_offence",
  "offence_no_card",
  "offence_yellow",
  "offence_red",
] as const

export const actionTypeLabels = [
  "standing_tackle",
  "tackle",
  "holding",
  "pushing",
  "challenge",
  "dive",
  "high_leg",
  "elbowing",
  "unknown",
] as const

export const supportedScopes = ["foul_review_context"] as const

export const unsupportedScopes = [
  "offside",
  "handball",
  "penalty_no_penalty",
  "mistaken_identity",
] as const

export type SanctionLabel = (typeof sanctionLabels)[number]
export type ActionTypeLabel = (typeof actionTypeLabels)[number]
export type SupportedScope = (typeof supportedScopes)[number]
export type UnsupportedScope = (typeof unsupportedScopes)[number]
export type AnalysisScope = SupportedScope | UnsupportedScope

export type VarReviewCategory =
  | "not_applicable"
  | "direct_red_related"
  | "unsupported_in_v1"

export type ErrorCode =
  | "invalid_request"
  | "clip_too_large"
  | "unsupported_media_type"
  | "clip_too_long"
  | "unsupported_scope"
  | "rate_limited"
  | "inference_failed"
  | "model_unavailable"

export interface PredictionAlternative<TLabel extends string> {
  label: TLabel
  confidence: number
}

export interface Prediction<TLabel extends string> {
  label: TLabel
  confidence: number
  alternatives: PredictionAlternative<TLabel>[]
}

export interface AnalyzeResponse {
  spec_version: typeof SPEC_VERSION
  model_version: string
  request_id: string
  status: "ok"
  sanction_prediction: Prediction<SanctionLabel>
  action_type_prediction: Prediction<ActionTypeLabel>
  viewer_focus: string[]
  explanation: string
  limitations: string[]
  review_context: {
    var_review_category: VarReviewCategory
    official_decision_claimed: false
  }
  provenance: {
    dataset_family: string
    rules_source: string
    frames_sampled: number
  }
}

export interface ErrorResponse {
  spec_version: typeof SPEC_VERSION
  request_id: string
  status: "error"
  error: {
    code: ErrorCode
    message: string
    details?: Record<string, string | number | boolean | null>
  }
}

export type AnalyzeEnvelope = AnalyzeResponse | ErrorResponse

export function isErrorResponse(value: AnalyzeEnvelope): value is ErrorResponse {
  return value.status === "error"
}

export function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`
}
