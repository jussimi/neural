import { Network } from "../Network";
import { DatasetItem, Dataset } from "../Dataset";
import { GradientDescentOptimizer } from "../Optimizer";

const runXOR = () => {
  const dataSet: DatasetItem[] = [
    [[1, 1], 0],
    [[1, 0], 1],
    [[0, 1], 1],
    [[0, 0], 0],
  ].map((row) => {
    return { input: row[0], output: [row[1]] } as DatasetItem;
  });

  const w1 = [
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
  ];
  const w2 = [[-0.5, 0.5, 0.5]];

  const network = new Network(new Dataset(dataSet), null, "log-loss");
  network.add("tanh", 2);
  network.add("sigmoid", 1);

  const gd = new GradientDescentOptimizer(network, {
    learningRate: 1,
    epochs: 500 / dataSet.length,
    afterIteration: (_, { loss }, i) => {
      if (i % 20 === 0) {
        console.log("Iteration: ", i);
        console.log("   - loss: ", loss);
      }
    },
    stopCondition: (_, { loss }, i) => {
      const stop = loss < 0.01;
      if (stop) {
        console.log(" STOPPING: ", i);
        console.log("   - loss: ", loss);
      }
      return stop;
    },
  });

  network.initialize([w1, w2]);
  console.time();
  gd.optimize(false);
  console.timeEnd();

  console.log("\n");

  const { gradient, loss, results } = network.computeGradient();

  results.forEach((res) => {
    console.log("input: ", res.input[0]);
    console.log("output: ", res.output[0]);
    console.log("estimate: ", res.estimate.transpose().toArrays());
    console.log("loss: ", res.loss);
    console.log("\n");
  });

  console.log("Total loss: ", loss);
  console.log("Weights:");
  network.layers.map((r, i) => console.table(r.weights.toArrays()));
};

export default runXOR;
