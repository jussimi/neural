import { Matrix } from "./Matrix";
import { Network } from "./Network";

type OptimizationState = { loss: number; gradients: Matrix[] };
type OptimizerOptionFN<T> = (
  network: Network,
  payload: OptimizationState,
  iteration: number
) => T;

export type OptimizerOptions = {
  learningRate: LearningRate;
  maxIterations: number;
  beforeIteration?: (network: Network, iteration: number) => void;
  afterIteration?: OptimizerOptionFN<void>;
  stopCondition?: OptimizerOptionFN<boolean>;
  afterAll?: () => void;
};

export type Optimizer = {
  optimize: (
    network: Network,
    computeGradients: () => OptimizationState
  ) => void;
} & OptimizerOptions;

type LearningRate = number | ((iteration: number) => number);

class BaseOptimizer implements Optimizer {
  learningRate: LearningRate;
  maxIterations: number;

  beforeIteration: OptimizerOptions["beforeIteration"];
  afterIteration: OptimizerOptions["afterIteration"];
  stopCondition: OptimizerOptions["stopCondition"];
  afterAll?: OptimizerOptions["afterAll"];

  doUpdate: (
    learningRate: number,
    data: OptimizationState,
    network: Network
  ) => void;

  constructor(options: OptimizerOptions) {
    this.learningRate = options.learningRate;
    this.maxIterations = options.maxIterations;

    this.beforeIteration = options.beforeIteration;
    this.afterIteration = options.afterIteration;
    this.stopCondition = options.stopCondition;
    this.afterAll = options.afterAll;
    this.doUpdate = () => {};
  }

  optimize: Optimizer["optimize"] = (network: Network, computeGradients) => {
    for (let i = 0; i < this.maxIterations; i += 1) {
      this.beforeIteration?.(network, i);

      const data = computeGradients();

      if (this.stopCondition?.(network, data, i)) {
        break;
      }

      const lr =
        typeof this.learningRate === "function"
          ? this.learningRate(i)
          : this.learningRate;

      if (lr < 0) {
        throw new Error("negative learningRate!");
      }

      this.doUpdate(lr, data, network);

      this.afterIteration?.(network, data, i);
    }

    this.afterAll?.();
  };
}

export class GradientDescentOptimizer extends BaseOptimizer {
  velocity: Matrix[] = [];
  momentum: number;

  currentWeights: Matrix[] = [];

  constructor(
    options: OptimizerOptions & { momentum?: number; nesterov?: boolean }
  ) {
    super(options);

    this.momentum = options.momentum || 0;

    if (options.nesterov) {
      this.beforeIteration = (network, iteration) => {
        options.beforeIteration?.(network, iteration);
        this.currentWeights = network.layers.map((l) => l.weights);

        if (this.velocity.length) {
          network.layers.forEach((l, i) => {
            l.weights = l.weights.sum(this.velocity[i].scale(this.momentum));
          });
        }
      };
    }

    this.doUpdate = (learningRate, data, network) => {
      const { gradients } = data;
      if (!this.velocity.length) {
        this.initializeVelocity(gradients);
      }

      this.velocity = this.velocity.map((vel, i) => {
        return vel
          .scale(this.momentum)
          .subtract(gradients[i].scale(learningRate));
      });

      network.layers.forEach((layer, i) => {
        let weights: Matrix;
        if (options.nesterov) {
          weights = this.currentWeights[i];
        } else {
          weights = layer.weights;
        }
        layer.updateWeights(weights.sum(this.velocity[i]));
      });
    };
  }

  private initializeVelocity(gradients: Matrix[]) {
    this.velocity = gradients.map((grad) => {
      return grad.scale(0);
    });
  }
}
