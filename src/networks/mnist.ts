import { TanH, Sigmoid, ReLu, Softmax } from "../activation";
import { CELoss, LogLoss, MSELoss } from "../error";
import { Layer } from "../Layer";
import { Matrix } from "../Matrix";
import { DataSet, Network } from "../Network";
import { GradientDescentOptimizer } from "../Optimizer";

import fs from "fs";

const oneHotEncode = (label: number, len: number): number[] => {
  const vector = [...Array(len).keys()].map((x) => 0);
  vector[label] = 1;
  return vector;
};

const readDataSet = (path: string) => {
  return fs
    .readFileSync(path)
    .toString()
    .split("\n")
    .filter((x) => !!x);
};

const runMnist = () => {
  const batchSize = 100;
  const labelSize = 10;

  const parseLine = (line: string): DataSet[1] => {
    const items = line.split(",").map((i) => parseInt(i));
    const label = oneHotEncode(items.shift() as number, labelSize);
    const point = items;
    return [point, label];
  };

  const trainData = readDataSet("./mnist_train.csv");
  const testData = readDataSet("./mnist_test.csv").map((x) => parseLine(x));

  const generateBatch = (): DataSet => {
    const set: DataSet = [];
    const usedIndices = new Map<number, number>();

    while (set.length < batchSize) {
      inner: while (true) {
        const nextIndex = Math.floor(Math.random() * trainData.length);
        if (usedIndices.has(nextIndex)) {
          continue;
        }
        usedIndices.set(nextIndex, 1);
        const data = parseLine(trainData[nextIndex]);
        if (data[0].length !== 784) throw new Error("dfd");
        set.push(data);
        break inner;
      }
    }
    return set;
  };

  const schedule = (i: number) => {
    const T = 600;
    const e0 = 1;
    const eT = 0.1;
    const alpha = i / T;
    return i >= T ? eT : (1 - alpha) * e0 + alpha * eT;
  };

  const times: number[] = [];

  const startTime = Date.now();
  let now = Date.now();
  const gd = new GradientDescentOptimizer(schedule, 6000, {
    beforeIteration: () => {
      now = Date.now();
    },
    afterIteration(network, { gradients, loss }, i) {
      const batch = generateBatch();
      network.setData(batch);
      if (i % 100 === 0) {
        console.log("Iteration: ", i);
        console.log("   - loss: ", loss);
        console.log("   - took: ", `${Date.now() - now} ms`);
        console.log("  - taken: ", `${(Date.now() - startTime) / 1000} s`);
      }
      if (i > 0 && i % 600 === 0) {
        let totalLoss = 0;
        let correctCount = 0;
        testData.forEach((point) => {
          const { loss, estimate } = network.validate(point);
          totalLoss += loss / testData.length;
          const label = point[1].indexOf(1);
          const values = estimate.transpose().toArrays()[0];
          const max = Math.max(...values);
          if (label !== -1 && label === values.indexOf(max)) {
            correctCount += 1;
          }
        });
        console.log("\nValidation loss: %d", totalLoss);
        console.log("      percentage: %d", correctCount / testData.length);
      }
      times.push(Date.now() - now);
    },
    stopCondition: (_, { loss }) => {
      return loss < 0.01 || isNaN(loss);
    },
    afterAll: () => {
      console.log(
        "Avg",
        times.reduce((avg, x) => avg + x / times.length, 0)
      );
    },
  });

  const l2 = new Layer(Sigmoid, 32);
  const l3 = new Layer(Softmax, labelSize);

  const network = new Network(generateBatch(), CELoss, gd, [l2, l3]);

  network.initialize();
  network.optimize();

  // console.log("Total loss: ", loss);
};

export default runMnist;
