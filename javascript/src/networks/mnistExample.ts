import fs from "fs";

import { Network } from "../Network";
import { AdamOptimizer } from "../Optimizer";
import { Dataset } from "../Dataset";
import { oneHotEncode } from "../utils";

const LABEL_SIZE = 10;
const BATCH_SIZE = 200;
const EPOCHS = 40;

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

const network = new Network(trainData, testData, "cross-entropy");
network.add("relu", 32);
network.add("softmax", LABEL_SIZE);

const optimizer = new AdamOptimizer(network, {
  epochs: EPOCHS,
  momentDecay: 0.9,
  gradientDecay: 0.999,
  learningRate: 0.001,
});

const { results, took } = optimizer.optimize();
console.log("Took %s", took);
console.log("Accuracy: ", results[results.length - 1]);
