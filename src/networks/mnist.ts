import { DataSet, Network } from "../Network";
import {
  AdaGradOptimizer,
  GradientDescentOptimizer,
  OptimizerOptions,
} from "../Optimizer";

import fs from "fs";
import { oneHotEncode, generateBatch } from "../utils";

const LABEL_SIZE = 10;
const BATCH_SIZE = 200;
const EPOCHS = 10;

const readDataSet = (path: string): DataSet => {
  return fs
    .readFileSync(path)
    .toString()
    .split("\n")
    .filter((x) => !!x)
    .map((line: string) => {
      const items = line.split(",").map((i) => parseInt(i));
      const output = oneHotEncode(items.shift() as number, LABEL_SIZE);
      const input = items;
      return { input, output };
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
      const label = point.output.indexOf(1);
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
    const initial = 0.5;
    const decay = 0.1;
    const min = 0.05;
    return Math.max(min, initial * (1 / (1 + decay * i)));
  };

  const start = Date.now();
  const options: OptimizerOptions = {
    learningRate: schedule,
    maxIterations: iterations,
    afterIteration(network, { gradients, loss }, i) {
      const batch = generateBatch(trainData, BATCH_SIZE);
      network.setData(batch);
      if (i % 10 === 0) {
        console.log(
          `Iteration ${i}, time taken ${
            Math.round((Date.now() - start) / 10) / 100
          }s`
        );
        console.log("   - loss: ", loss);
      }
      if (i > 0 && i % epochIterations === 0) {
        validate(network);
      }
    },
    stopCondition: (_, { loss }) => {
      return loss < 0.01 || isNaN(loss);
    },
    afterAll: () => {
      validate(network);
    },
  };

  // const optimizer = new GradientDescentOptimizer({
  //   ...options,
  //   momentum: 0,
  //   nesterov: false,
  // });

  const optimizer = new AdaGradOptimizer({
    ...options,
    learningRate: 0.01,
  });

  const initialBatch = generateBatch(trainData, BATCH_SIZE);
  const network = new Network(initialBatch, "cross-entropy", optimizer)
    .add("sigmoid", 512)
    .add("softmax", LABEL_SIZE);

  network.train();

  // console.log("Total loss: ", loss);
};

export default runMnist;
