import { Network } from "../Network";
import { OptimizerOptions } from "../Optimizer";

import fs from "fs";
import { DatasetItem, Dataset } from "../Dataset";
import { runOptimizerComparison } from "./shared";
import { Smote } from "../utils";

const [headers, ...unparsedRows] = fs
  .readFileSync("creditcard.csv")
  .toString()
  .split("\n")
  .filter((r) => !!r)
  .map((r) => r.split(",").map((x) => JSON.parse(x)));

const rows = unparsedRows.map((x) => x.map((x) => parseFloat(x)));

const maxValues: Record<string, number> = {};

headers.forEach((h, i) => {
  let max = 0;

  for (const row of rows) {
    const value = row[i];
    const abs = Math.abs(value);
    if (abs > max) {
      max = abs;
    }
  }
  maxValues[h] = max;
});

const data: DatasetItem[] = rows.map((values) => {
  const label: number[] = [];
  const result: number[] = [];
  values.forEach((value, i) => {
    const header = headers[i];
    if (header === "Class") {
      label.push(value);
      return;
    }
    result.push(value / maxValues[header]);
  });
  return { input: result, output: label };
});

const isFraud = data.filter((d) => d.output[0] === 1);
const notFraud = data.filter((d) => d.output[0] === 0);

const split = 0.3;
const isFraudSplit = Dataset.createTrainAndTest(isFraud, 1, split);
const notFraudSplit = Dataset.createTrainAndTest(notFraud, 1, split);

const BATCH_SIZE = 50;
const EPOCHS = 3;

const oversampled = Smote(
  isFraudSplit.train.items.map((i) => i.input),
  10000,
  10
).map((x) => {
  const item: DatasetItem = {
    input: x,
    output: isFraud[0].output,
  };
  return item;
});

const trainItems = [...oversampled, ...notFraudSplit.train.items];

const trainData = new Dataset(trainItems, BATCH_SIZE);

const testData = new Dataset([
  ...isFraudSplit.test.items,
  ...notFraudSplit.test.items,
]);

const schedule = (i: number) => {
  const initial = 0.5;
  const decay = 0.1;
  const min = 0.05;
  return Math.max(min, initial * (1 / (1 + decay * i)));
};

const runFraud = () => {
  const network = new Network(trainData, testData, "log-loss");
  network.add("relu", 32);
  network.add("sigmoid", 2);
  network.add("sigmoid", 1);

  const options: OptimizerOptions = {
    learningRate: schedule,
    epochs: EPOCHS,
  };

  runOptimizerComparison(network, options, null);

  console.log(network.validateDataset(isFraudSplit.test.items));
  console.log(network.validateDataset(notFraudSplit.test.items));
};

export default runFraud;
