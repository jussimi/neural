import { Matrix } from "./Matrix";
import { Network } from "./Network";

type OptimizationState = { loss: number; gradients: Matrix[] };
type OptimizerOptionFN<T> = (
  network: Network,
  payload: OptimizationState,
  iteration: number
) => T;

type OptimizerOptions = {
  beforeIteration?: (network: Network, iteration: number) => void;
  afterIteration?: OptimizerOptionFN<void>;
  stopCondition?: OptimizerOptionFN<boolean>;
};

export type Optimizer = {
  optimize: (
    network: Network,
    computeGradients: () => OptimizationState
  ) => void;
} & OptimizerOptions;

type LearningRate = number | ((iteration: number) => number);
export class GradientDescentOptimizer implements Optimizer {
  learningRate: LearningRate;
  iterations: number;

  beforeIteration: OptimizerOptions["beforeIteration"];
  afterIteration: OptimizerOptions["afterIteration"];
  stopCondition: OptimizerOptions["stopCondition"];

  constructor(
    learningRate: LearningRate,
    iterations: number,
    options: OptimizerOptions = {}
  ) {
    this.learningRate = learningRate;
    this.iterations = iterations;

    this.beforeIteration = options.beforeIteration;
    this.afterIteration = options.afterIteration;
    this.stopCondition = options.stopCondition;
  }

  optimize: Optimizer["optimize"] = (network: Network, computeGradients) => {
    for (let i = 0; i < this.iterations; i += 1) {
      this.beforeIteration?.(network, i);

      const data = computeGradients();
      const { gradients, loss } = data;

      if (this.stopCondition?.(network, data, i)) {
        break;
      }

      const lr =
        typeof this.learningRate === "function"
          ? this.learningRate(i)
          : this.learningRate;
      network.layers.forEach((layer, i) => {
        const weights = layer.weights.sum(gradients[i].scale(-lr));
        layer.weights = weights;
      });

      this.afterIteration?.(network, data, i);
    }
  };
}
