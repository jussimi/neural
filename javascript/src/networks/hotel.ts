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
import { oneHotEncode, generateBatch, ValidationRes } from "../utils";

const BATCH_SIZE = 100;
const EPOCHS = 30;

const schedule = (i: number) => {
  const initial = 0.5;
  const decay = 0.1;
  const min = 0.05;
  return Math.max(min, initial * (1 / (1 + decay * i)));
};

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

const data: DataSet = rows
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

const splitIndex = Math.ceil(0.3 * data.length);
const testData = data.slice(0, splitIndex);
const trainData = data.slice(splitIndex);

const setSize = trainData.length;
const epochIterations = Math.floor(setSize / BATCH_SIZE);

const iterations = epochIterations * EPOCHS;

const validate = (nn: Network, data: DataSet): ValidationRes => {
  let totalLoss = 0;
  let correctCount = 0;

  data.forEach((point) => {
    const { loss, estimate } = nn.validate(point);
    totalLoss += loss;
    const result = Math.round(estimate.get(0, 0));
    if (result === point.output[0]) {
      correctCount += 1;
    }
  });
  return {
    loss: totalLoss / data.length,
    percentage: correctCount / data.length,
  };
};

const runHotel = () => {
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
        console.log(result);
      }
    },
    stopCondition: (_, { loss }) => {
      return isNaN(loss);
    },
  };
  const initialBatch = generateBatch(trainData, BATCH_SIZE);

  const network = new Network(initialBatch, "log-loss");
  network.add("relu", 128);
  network.add("sigmoid", 1);

  const optimizers: { name: string; optimizer: Optimizer }[] = [
    {
      name: "sgd",
      optimizer: new GradientDescentOptimizer(network, {
        ...options,
        momentum: 0,
        nesterov: false,
      }),
    },
    {
      name: "momentum",
      optimizer: new GradientDescentOptimizer(network, {
        ...options,
        momentum: 0.5,
        nesterov: false,
      }),
    },
    {
      name: "nesterov",
      optimizer: new GradientDescentOptimizer(network, {
        ...options,
        momentum: 0.5,
        nesterov: true,
      }),
    },
    {
      name: "AdaGrad",
      optimizer: new AdaGradOptimizer(network, {
        ...options,
        learningRate: 0.01,
      }),
    },
    {
      name: "AdaDelta",
      optimizer: new AdaDeltaOptimizer(network, {
        ...options,
        decay: 0.9,
      }),
    },
    {
      name: "RMSProp",
      optimizer: new RMSPropOptimizer(network, {
        ...options,
        decay: 0.9,
        learningRate: 0.001,
      }),
    },
    {
      name: "Adam",
      optimizer: new AdamOptimizer(network, {
        ...options,
        momentDecay: 0.9,
        gradientDecay: 0.999,
        learningRate: 0.001,
      }),
    },
  ];

  for (const { name, optimizer } of optimizers) {
    console.log("Start %s", name);
    const start = Date.now();

    network.initialize();
    optimizer.optimize();

    const took = (Date.now() - start) / 1000;
    console.log("finished %s. Took %d", name, took);
  }
};

export default runHotel;
