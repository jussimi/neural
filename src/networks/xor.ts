import { DataSet, Network } from "../Network";
import { GradientDescentOptimizer } from "../Optimizer";

const runXOR = () => {
  const dataSet: DataSet = [
    [[1, 1], 0],
    [[1, 0], 1],
    [[0, 1], 1],
    [[0, 0], 0],
  ].map((row) => {
    const [input, output] = row;
    return [input as number[], [output as number]];
  });

  const w1 = [
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
  ];
  const w2 = [[-0.5, 0.5, 0.5]];

  const gd = new GradientDescentOptimizer(1, 500, {
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

  const network = new Network(dataSet, "log-loss", gd)
    .add("tanh", 2)
    .add("sigmoid", 1);

  network.initialize([w1, w2]);
  network.optimize();

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
