import { TanH, Sigmoid, ReLu } from "../activation";
import { LogLoss, MSELoss } from "../error";
import { Layer } from "../Layer";
import { Matrix } from "../Matrix";
import { DataSet, Network } from "../Network";
import { GradientDescentOptimizer } from "../Optimizer";

import fs from "fs";
import path from "path";

const oneHotEncode = (label: number, len: number): Matrix => {
  const vector = [...Array(len).keys()].map((x) => 0);
  vector[label] = 1;
  return Matrix.fromList(vector);
};

const runMnist = () => {
  const trainData = fs
    .readFileSync(path.join(__dirname, "../../mnist_train.csv"))
    .toString()
    .split("\n");

  const batchSize = 200;
  const labelSize = 10;

  const generateBatch = (i: number): DataSet => {
    return trainData
      .slice(i * batchSize, i * batchSize + batchSize)
      .map((row) => {
        const items = row.split(",").map((i) => parseInt(i));
        const label = [items.shift() as number]; //oneHotEncode(items.shift() as number, labelSize);
        const point = items;
        return [point, label] as [number[], number[]];
      });
  };

  let now = Date.now();
  const gd = new GradientDescentOptimizer(1, 10, {
    beforeIteration: () => {
      now = Date.now();
    },
    afterIteration(network, { loss }, i) {
      const batch = generateBatch(i);
      network.setData(batch);
      console.log("Iteration: ", i);
      console.log("   - loss: ", loss);
      console.log(Date.now() - now);
    },
    stopCondition: (_, { loss }) => {
      return loss < 0.01;
    },
  });

  const l1 = new Layer(ReLu, 512);
  const l2 = new Layer(ReLu, 512);
  const l3 = new Layer(Sigmoid, 1);

  const network = new Network(generateBatch(0), MSELoss, gd, [l1, l2, l3]);

  network.initialize();
  network.optimize();

  // console.log("Total loss: ", loss);
};

export default runMnist;
