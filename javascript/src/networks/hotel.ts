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

  const isPossiblyCategorical = values.length < 40 && !h.startsWith("no_");
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

const data: DatasetItem[] = rows
  .sort(() => Math.random() - 0.5)
  .map((values) => {
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

const BATCH_SIZE = 100;
const EPOCHS = 30;

const { train: trainData, test: testData } = Dataset.createTrainAndTest(
  data,
  BATCH_SIZE,
  0.3
);

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
  network.add("relu", 128);
  network.add("relu", 64);
  network.add("sigmoid", 32);
  network.add("sigmoid", 1);

  runOptimizerComparison(network, options, "hotel");
};

export default runHotel;
