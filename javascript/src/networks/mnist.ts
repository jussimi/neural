import { Network } from "../Network";
import { OptimizerOptions } from "../Optimizer";

import fs from "fs";
import { Dataset } from "../Dataset";
import { oneHotEncode } from "../utils";
import { runOptimizerComparison } from "./shared";

const LABEL_SIZE = 10;
const BATCH_SIZE = 200;
const EPOCHS = 20;

const readDataSet = (
  path: string,
  batchSize: number | undefined = undefined
): Dataset => {
  const items = fs
    .readFileSync(path)
    .toString()
    .split("\n")
    .filter((x) => !!x)
    .map((line: string) => {
      const items = line.split(",").map((i) => parseInt(i));
      const output = oneHotEncode(items.shift() as number, LABEL_SIZE);
      const input = items.map((i) => i / 255); // Scale values to 0-1.
      return { input, output };
    });
  return new Dataset(items, batchSize);
};

const trainData = readDataSet("./mnist_train.csv", BATCH_SIZE);
const testData = readDataSet("./mnist_test.csv");

const schedule = (i: number) => {
  const initial = 0.5;
  const decay = 0.1;
  const min = 0.05;
  return Math.max(min, initial * (1 / (1 + decay * i)));
};

const runMnist = () => {
  const options: OptimizerOptions = {
    learningRate: schedule,
    epochs: EPOCHS,
  };

  const network = new Network(trainData, testData, "cross-entropy");
  network.add("relu", 128);
  network.add("relu", 64);
  network.add("relu", 32);
  network.add("softmax", LABEL_SIZE);

  runOptimizerComparison(network, options, "mnist");
};

export default runMnist;
