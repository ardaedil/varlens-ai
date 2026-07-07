import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import test from "node:test"

const schemaPath = path.resolve("../../packages/contracts/analyze.schema.json")
const labelsPath = path.resolve("../../config/labels.json")

test("analyze schema keeps v1 honesty fields required", () => {
  const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"))
  const required = schema.$defs.analyzeResponse.required

  assert.equal(schema.$defs.reviewContext.properties.official_decision_claimed.const, false)
  assert.ok(required.includes("limitations"))
  assert.ok(required.includes("review_context"))
})

test("configured labels match the schema enums", () => {
  const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"))
  const labels = JSON.parse(fs.readFileSync(labelsPath, "utf8"))

  assert.deepEqual(
    labels.sanction_labels.map((label) => label.id).sort(),
    schema.$defs.sanctionLabel.enum.toSorted(),
  )
  assert.deepEqual(
    labels.action_type_labels.map((label) => label.id).sort(),
    schema.$defs.actionTypeLabel.enum.toSorted(),
  )
})
