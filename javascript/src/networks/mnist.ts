import { DataSet, Network } from "../Network";
import {
  AdaDeltaOptimizer,
  AdaGradOptimizer,
  AdamOptimizer,
  GradientDescentOptimizer,
  Optimizer,
  OptimizerOptions,
  RMSPropOptimizer,
} from "../Optimizer";

import fs from "fs";
import { oneHotEncode, generateBatch, isCorrectCategory } from "../utils";

const LABEL_SIZE = 10;
const BATCH_SIZE = 200;
const EPOCHS = 1;

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

type ValidationRes = {
  loss: number;
  percentage: number;
};
type Result = {
  iteration: number;
  train: ValidationRes;
  test: ValidationRes;
};

const validate = (nn: Network, data: DataSet): ValidationRes => {
  let totalLoss = 0;
  let correctCount = 0;
  data.forEach((point) => {
    const { loss, estimate } = nn.validate(point);
    totalLoss += loss;
    if (isCorrectCategory(estimate, point.output)) {
      correctCount += 1;
    }
  });
  return {
    loss: totalLoss / data.length,
    percentage: correctCount / data.length,
  };
};

const schedule = (i: number) => {
  const initial = 0.5;
  const decay = 0.1;
  const min = 0.05;
  return Math.max(min, initial * (1 / (1 + decay * i)));
};

const runMnist = () => {
  let currentTrainingResults: Result[] = [];

  const options: OptimizerOptions = {
    learningRate: schedule,
    maxIterations: iterations,
    afterIteration(network, { loss }, i) {
      const batch = generateBatch(trainData, BATCH_SIZE);
      network.setData(batch);
      if (i % epochIterations === 0 || i === iterations) {
        const testRes = validate(network, testData);
        const trainRes = validate(network, trainData);
        const result = {
          iteration: i,
          train: trainRes,
          test: testRes,
        };
        currentTrainingResults.push(result);
        console.log(result);
      }
    },
    stopCondition: (_, { loss }) => {
      return loss < 0.01 || isNaN(loss);
    },
  };

  const optimizers: { name: string; optimizer: Optimizer }[] = [
    {
      name: "sgd",
      optimizer: new GradientDescentOptimizer({
        ...options,
        momentum: 0,
        nesterov: false,
      }),
    },
    {
      name: "momentum",
      optimizer: new GradientDescentOptimizer({
        ...options,
        momentum: 0.5,
        nesterov: false,
      }),
    },
    {
      name: "nesterov",
      optimizer: new GradientDescentOptimizer({
        ...options,
        momentum: 0.5,
        nesterov: true,
      }),
    },
    {
      name: "AdaGrad",
      optimizer: new AdaGradOptimizer({
        ...options,
        learningRate: 0.01,
      }),
    },
    {
      name: "AdaDelta",
      optimizer: new AdaDeltaOptimizer({
        ...options,
        decay: 0.9,
      }),
    },
    {
      name: "RMSProp",
      optimizer: new RMSPropOptimizer({
        ...options,
        decay: 0.9,
        learningRate: 0.001,
      }),
    },
    {
      name: "Adam",
      optimizer: new AdamOptimizer({
        ...options,
        momentDecay: 0.9,
        gradientDecay: 0.999,
        learningRate: 0.001,
      }),
    },
  ];

  const results: { optimizer: string; took: number; results: Result[] }[] = [];

  for (const { name, optimizer } of optimizers) {
    console.log("Start %s", name);
    const start = Date.now();
    const initialBatch = generateBatch(trainData, BATCH_SIZE);
    const network = new Network(initialBatch, "cross-entropy", optimizer)
      .add("sigmoid", 512)
      .add("softmax", LABEL_SIZE);

    network.train();

    const took = (Date.now() - start) / 1000;
    console.log("finished %s. Took %d", name, took);
    results.push({ optimizer: name, results: currentTrainingResults, took });
    currentTrainingResults = [];
  }

  for (const value of results) {
    let min = value.results[0];
    for (const res of value.results) {
      if (res.test.loss < min.test.loss) {
        min = res;
      }
    }
    console.log(value.optimizer, min);
  }

  const fileName = `results-${new Date().toISOString()}.json`;
  fs.writeFileSync(fileName, JSON.stringify(results));
};

export default runMnist;
