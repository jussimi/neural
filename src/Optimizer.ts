import { Matrix } from "./Matrix";
import { Layer } from "./Layer";

export interface Optimizer {
  optimize: (
    layers: Layer[],
    computeGradients: () => { loss: number; gradients: Matrix[] }
  ) => void;
}

type LearningRate = number | ((iteration: number) => number);
export class GradientDescentOptimizer implements Optimizer {
  learningRate: LearningRate;
  iterations: number;

  constructor(learningRate: LearningRate, iterations: number) {
    this.learningRate = learningRate;
    this.iterations = iterations;
  }

  optimize: Optimizer["optimize"] = (layers: Layer[], computeGradients) => {
    for (let i = 0; i < this.iterations; i += 1) {
      const { gradients, loss } = computeGradients();

      console.log("Iteration: ", i);
      console.log("   - loss: ", loss);

      const lr =
        typeof this.learningRate === "function"
          ? this.learningRate(i)
          : this.learningRate;
      layers.forEach((layer, i) => {
        const weights = layer.weights.sum(gradients[i].scale(-lr));
        layer.weights = weights;
      });
    }
    console.log("\n");
  };
}
