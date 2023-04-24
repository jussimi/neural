import { Network } from "../Network";
import {
  AdaDeltaOptimizer,
  AdaGradOptimizer,
  AdamOptimizer,
  EpochResult,
  GradientDescentOptimizer,
  Optimizer,
  OptimizerOptions,
  RMSPropOptimizer,
} from "../Optimizer";

import fs from "fs";

export const runOptimizerComparison = (
  network: Network,
  optimizerOpts: OptimizerOptions,
  name: string | null
) => {
  const optimizers: { name: string; optimizer: Optimizer }[] = [
    // {
    //   name: "sgd",
    //   optimizer: new GradientDescentOptimizer(network, {
    //     ...optimizerOpts,
    //     momentum: 0,
    //     nesterov: false,
    //   }),
    // },
    // {
    //   name: "momentum",
    //   optimizer: new GradientDescentOptimizer(network, {
    //     ...optimizerOpts,
    //     momentum: 0.5,
    //     nesterov: false,
    //   }),
    // },
    // {
    //   name: "nesterov",
    //   optimizer: new GradientDescentOptimizer(network, {
    //     ...optimizerOpts,
    //     momentum: 0.5,
    //     nesterov: true,
    //   }),
    // },
    // {
    //   name: "AdaGrad",
    //   optimizer: new AdaGradOptimizer(network, {
    //     ...optimizerOpts,
    //     learningRate: 0.01,
    //   }),
    // },
    // {
    //   name: "AdaDelta",
    //   optimizer: new AdaDeltaOptimizer(network, {
    //     ...optimizerOpts,
    //     decay: 0.9,
    //   }),
    // },
    // {
    //   name: "RMSProp",
    //   optimizer: new RMSPropOptimizer(network, {
    //     ...optimizerOpts,
    //     decay: 0.9,
    //     learningRate: 0.001,
    //   }),
    // },
    {
      name: "Adam",
      optimizer: new AdamOptimizer(network, {
        ...optimizerOpts,
        momentDecay: 0.9,
        gradientDecay: 0.999,
        learningRate: 0.001,
      }),
    },
  ];

  const results: {
    optimizer: string;
    took: number;
    results: EpochResult[];
  }[] = [];

  for (const { name, optimizer } of optimizers) {
    console.log("Start %s", name);

    const res = optimizer.optimize();

    console.log("finished %s. Took %d", name, res.took);
    results.push({ optimizer: name, ...res });
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

  if (name !== null) {
    const fileName = `${name}-${new Date().toISOString()}.json`;
    fs.writeFileSync(fileName, JSON.stringify(results));
  }
};
