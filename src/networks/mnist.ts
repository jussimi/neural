import { DataSet, Network } from "../Network";
import { GradientDescentOptimizer } from "../Optimizer";

import fs from "fs";
import { oneHotEncode, generateBatch } from "../utils";

const LABEL_SIZE = 10;
const BATCH_SIZE = 128;
const EPOCHS = 10;

const readDataSet = (path: string): DataSet => {
  return fs
    .readFileSync(path)
    .toString()
    .split("\n")
    .filter((x) => !!x)
    .map((line: string) => {
      const items = line.split(",").map((i) => parseInt(i));
      const label = oneHotEncode(items.shift() as number, LABEL_SIZE);
      const point = items;
      return [point, label];
    });
};

const trainData = readDataSet("./mnist_train.csv");
const testData = readDataSet("./mnist_test.csv");

const setSize = trainData.length;
const epochIterations = Math.floor(setSize / BATCH_SIZE);

const iterations = epochIterations * EPOCHS;

const runMnist = () => {
  const validate = (nn: Network) => {
    let totalLoss = 0;
    let correctCount = 0;
    testData.forEach((point) => {
      const { loss, estimate } = nn.validate(point);
      totalLoss += loss / testData.length;
      const label = point[1].indexOf(1);
      const values = estimate.transpose().toArrays()[0];
      const max = Math.max(...values);
      if (label !== -1 && label === values.indexOf(max)) {
        correctCount += 1;
      }
    });
    console.log("\nValidation loss: %d", totalLoss);
    console.log("     percentage: %d", correctCount / testData.length);
  };

  const schedule = (i: number) => {
    const initial = 1;
    const decay = 0.1;
    const min = 0.01;
    return Math.max(min, initial * (1 / (1 + decay * i)));
  };

  const times: number[] = [];

  const startTime = Date.now();
  let now = Date.now();
  const gd = new GradientDescentOptimizer(schedule, iterations, {
    beforeIteration: () => {
      now = Date.now();
    },
    afterIteration(network, { gradients, loss }, i) {
      const batch = generateBatch(trainData, BATCH_SIZE);
      network.setData(batch);
      if (i % 100 === 0) {
        console.log("Iteration: ", i);
        console.log("   - loss: ", loss);
        console.log("   - took: ", `${Date.now() - now} ms`);
        console.log("  - taken: ", `${(Date.now() - startTime) / 1000} s`);
      }
      if (i > 0 && i % epochIterations === 0) {
        validate(network);
      }
      times.push(Date.now() - now);
    },
    stopCondition: (_, { loss }) => {
      return loss < 0.01 || isNaN(loss);
    },
    afterAll: () => {
      console.log(
        "Average time per iteration",
        times.reduce((avg, x) => avg + x / times.length, 0)
      );

      validate(network);
    },
  });

  const initialBatch = generateBatch(trainData, BATCH_SIZE);
  const network = new Network(initialBatch, "cross-entropy", gd)
    .add("sigmoid", 32)
    .add("softmax", LABEL_SIZE);

  network.initialize();
  network.optimize();

  // console.log("Total loss: ", loss);
};

export default runMnist;
