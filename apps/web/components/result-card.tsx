import type {
  ActionTypeLabel,
  AnalyzeResponse,
  Prediction,
  SanctionLabel,
} from "@varlens/contracts"
import { formatConfidence } from "@varlens/contracts"
import { PredictionMeter } from "@/components/prediction-meter"
import { StatusBanner } from "@/components/status-banner"

const labelText: Record<SanctionLabel | ActionTypeLabel, string> = {
  no_offence: "No offence",
  offence_no_card: "Offence, no card",
  offence_yellow: "Offence, yellow card",
  offence_red: "Offence, red card",
  standing_tackle: "Standing tackle",
  tackle: "Tackle",
  holding: "Holding",
  pushing: "Pushing",
  challenge: "Challenge",
  dive: "Dive",
  high_leg: "High leg",
  elbowing: "Elbowing",
  unknown: "Unknown",
}

interface ResultCardProps {
  result: AnalyzeResponse
}

export function ResultCard({ result }: ResultCardProps) {
  const lowConfidence =
    result.sanction_prediction.confidence < 0.7 || result.action_type_prediction.confidence < 0.7

  return (
    <div className="result-layout">
      {lowConfidence ? (
        <StatusBanner tone="warning" title="moderate confidence">
          Nearby labels remain plausible for this clip.
        </StatusBanner>
      ) : (
        <StatusBanner tone="neutral" title="analysis complete">
          The response is contract-compliant and includes limitations.
        </StatusBanner>
      )}

      <div className="prediction-grid">
        <PredictionBlock title="Sanction" prediction={result.sanction_prediction} />
        <PredictionBlock title="Action type" prediction={result.action_type_prediction} />
      </div>

      <p className="analysis-copy">{result.explanation}</p>

      <section>
        <h3>Viewer focus</h3>
        <ul className="focus-list">
          {result.viewer_focus.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section>
        <h3>Limitations</h3>
        <ul className="limitations-list">
          {result.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <div className="meta-grid">
        <div className="meta-item">
          <span>Model</span>
          <strong>{result.model_version}</strong>
        </div>
        <div className="meta-item">
          <span>Rules</span>
          <strong>{result.provenance.rules_source}</strong>
        </div>
        <div className="meta-item">
          <span>Request</span>
          <strong>{result.request_id}</strong>
        </div>
      </div>
    </div>
  )
}

function PredictionBlock<TLabel extends SanctionLabel | ActionTypeLabel>({
  title,
  prediction,
}: {
  title: string
  prediction: Prediction<TLabel>
}) {
  return (
    <article className="prediction-card">
      <header>
        <div>
          <h3>{title}</h3>
          <strong>{labelText[prediction.label]}</strong>
        </div>
        <PredictionMeter confidence={prediction.confidence} />
      </header>

      {prediction.alternatives.length ? (
        <ul className="alt-list">
          {prediction.alternatives.map((alternative) => (
            <li key={alternative.label}>
              {labelText[alternative.label]} - {formatConfidence(alternative.confidence)}
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  )
}
