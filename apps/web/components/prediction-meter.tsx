import { formatConfidence } from "@varlens/contracts"

interface PredictionMeterProps {
  confidence: number
}

export function PredictionMeter({ confidence }: PredictionMeterProps) {
  const percent = Math.max(0, Math.min(100, Math.round(confidence * 100)))

  return (
    <>
      <span className="confidence">{formatConfidence(confidence)}</span>
      <div className="meter" aria-label={`Confidence ${percent}%`}>
        <span style={{ width: `${percent}%` }} />
      </div>
    </>
  )
}
