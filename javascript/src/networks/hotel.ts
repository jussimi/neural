import { Network } from "../Network";
import { OptimizerOptions } from "../Optimizer";

import fs from "fs";
import { oneHotEncode } from "../utils";
import { DatasetItem, Dataset } from "../Dataset";
import { runOptimizerComparison } from "./shared";

const [headers, ...rows] = fs
  .readFileSync("reservations.csv")
  .toString()
  .split("\n")
  .filter((r) => !!r)
  .map((r) => r.split(","));

const distinctValues: Record<string, Map<string, number>> = {};
const maxValues: Record<string, number> = {};

headers.forEach((h, i) => {
  const values: string[] = [];

  for (const row of rows) {
    const value = row[i];
    if (!values.includes(value)) {
      values.push(value);
    }
  }

  const isPossiblyCategorical = values.length < 40;
  const firstValueAsNumber = parseFloat(values[0]);
  if (isPossiblyCategorical) {
    const labelMap = new Map();
    values.forEach((val, i) => labelMap.set(val, i));

    distinctValues[h] = labelMap;
  } else if (!isNaN(firstValueAsNumber)) {
    let max = 0;
    for (const val of values) {
      const parsed = parseFloat(val);
      if (parsed > max) {
        max = parsed;
      }
    }
    maxValues[h] = max;
  }
});

console.log(maxValues);
console.log(distinctValues);

const data: DatasetItem[] = rows.map((values) => {
  const label: number[] = [];
  const result: number[] = [];
  values.forEach((val, i) => {
    const header = headers[i];
    if (header === "booking_status") {
      const value = distinctValues[header].get(val);
      if (typeof value !== "number") throw new Error("invalid status");
      label.push(value);
      return;
    }
    const isNumerical = typeof maxValues[header] === "number";
    if (isNumerical) {
      result.push(parseFloat(val) / maxValues[header]);
      return;
    }
    const isCategorical = typeof distinctValues[header] !== "undefined";
    if (isCategorical) {
      const size = distinctValues[header].size;
      const value = distinctValues[header].get(val);
      if (typeof value !== "number") throw new Error("invalid value!");
      result.push(...oneHotEncode(value, size));
    }
  });
  return { input: result, output: label };
});

const isCancelled = data.filter((d) => d.output[0] === 1);
const notCancelled = data.filter((d) => d.output[0] === 0);

const split = 0.3;
const isCancelledSplit = Dataset.createTrainAndTest(isCancelled, 1, split);
const notCancelledSplit = Dataset.createTrainAndTest(notCancelled, 1, split);

const BATCH_SIZE = 50;
const EPOCHS = 20;

const trainItems = [
  ...isCancelledSplit.train.items,
  ...notCancelledSplit.train.items,
];

const trainData = new Dataset(trainItems, BATCH_SIZE);

const testData = new Dataset([
  ...isCancelledSplit.test.items,
  ...notCancelledSplit.test.items,
]);

const schedule = (i: number) => {
  const initial = 0.5;
  const decay = 0.1;
  const min = 0.05;
  return Math.max(min, initial * (1 / (1 + decay * i)));
};

const runHotel = () => {
  const options: OptimizerOptions = {
    learningRate: schedule,
    epochs: EPOCHS,
  };

  const network = new Network(trainData, testData, "log-loss");
  network.add("relu", 256);
  network.add("relu", 128);
  network.add("sigmoid", 1);

  runOptimizerComparison(network, options, "hotel");
};

export default runHotel;
