import { TanH, Sigmoid, LogLoss, ReLu } from "./src/functions";
import { Layer } from "./src/Layer";
import { Vector, Scalar, Matrix } from "./src/Matrix";
import { Network } from "./src/Network";
import { GradientDescentOptimizer } from "./src/Optimizer";

const dataSet: [Vector, Vector][] = [
  [[1, 1], 0],
  [[1, 0], 1],
  [[0, 1], 1],
  [[0, 0], 0],
].map((row) => {
  const [input, output] = row;
  return [new Vector(input as number[]), new Scalar(output as number)];
});

const w1 = new Matrix([
  [0.5, 0.5, 0.5],
  [-0.5, -0.5, -0.5],
]);
const w2 = new Matrix([[-0.5, 0.5, 0.5]]);

const l1 = new Layer(TanH, 2);
const l2 = new Layer(Sigmoid, 1);

const gd = new GradientDescentOptimizer(1, 500);
const network = new Network(dataSet, LogLoss, gd, [l1, l2]);

network.initialize([w1, w2]);
network.optimize();

const { gradient, loss, results } = network.computeGradient();

results.forEach((res) => {
  console.log("input: ", res.input.transpose().rows[0]);
  console.log("output: ", res.output.transpose().rows[0]);
  console.log("estimate: ", res.estimate.transpose().rows[0]);
  console.log("loss: ", res.loss);
  console.log("\n");
});

console.log("Total loss: ", loss);
console.log("Weights:");
network.layers.map((r, i) => console.table(r.weights.rows));
