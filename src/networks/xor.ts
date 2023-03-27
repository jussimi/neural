import { TanH, Sigmoid } from "../activation";
import { LogLoss } from "../error";
import { Layer } from "../Layer";
import { Vector, Scalar, Matrix, fromVect, transpose } from "../Matrix";
import { Network } from "../Network";
import { GradientDescentOptimizer } from "../Optimizer";

const runXOR = () => {
  const dataSet: [Vector, Vector][] = [
    [[1, 1], 0],
    [[1, 0], 1],
    [[0, 1], 1],
    [[0, 0], 0],
  ].map((row) => {
    const [input, output] = row;
    return [fromVect(input as number[]), [[output as number]]];
  });

  const w1 = [
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
  ];
  const w2 = [[-0.5, 0.5, 0.5]];

  const l1 = new Layer(TanH, 2);
  const l2 = new Layer(Sigmoid, 1);

  const gd = new GradientDescentOptimizer(1, 500, {
    beforeIteration: (_, { loss }, i) => {
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
  const network = new Network(dataSet, LogLoss, gd, [l1, l2]);

  network.initialize([w1, w2]);
  network.optimize();

  console.log("\n");

  const { gradient, loss, results } = network.computeGradient();

  results.forEach((res) => {
    console.log("input: ", transpose(res.input)[0]);
    console.log("output: ", transpose(res.output)[0]);
    console.log("estimate: ", transpose(res.estimate)[0]);
    console.log("loss: ", res.loss);
    console.log("\n");
  });

  console.log("Total loss: ", loss);
  console.log("Weights:");
  network.layers.map((r, i) => console.table(r.weights));
};

export default runXOR;
