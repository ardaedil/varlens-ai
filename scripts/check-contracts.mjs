import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const schemaPath = path.join(root, "packages", "contracts", "analyze.schema.json")
const labelsPath = path.join(root, "config", "labels.json")

const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"))
const labels = JSON.parse(fs.readFileSync(labelsPath, "utf8"))

const sanctionEnum = schema.$defs.sanctionLabel.enum
const actionEnum = schema.$defs.actionTypeLabel.enum
const configSanctions = labels.sanction_labels.map((label) => label.id)
const configActions = labels.action_type_labels.map((label) => label.id)

function sameMembers(left, right) {
  return (
    left.length === right.length &&
    [...left].sort().every((value, index) => value === [...right].sort()[index])
  )
}

const failures = []

if (!sameMembers(sanctionEnum, configSanctions)) {
  failures.push("Sanction labels differ between analyze.schema.json and config/labels.json")
}

if (!sameMembers(actionEnum, configActions)) {
  failures.push("Action labels differ between analyze.schema.json and config/labels.json")
}

if (!schema.$defs.analyzeResponse.required.includes("limitations")) {
  failures.push("Analyze response must require limitations")
}

if (schema.$defs.reviewContext.properties.official_decision_claimed.const !== false) {
  failures.push("The contract must forbid official decision claims")
}

if (failures.length) {
  console.error(failures.map((failure) => `- ${failure}`).join("\n"))
  process.exit(1)
}

console.log("Contract check passed")
