import { Layer } from "./Layer";
import { Matrix } from "./Matrix";
import { Network } from "./Network";
import { Result } from "./utils";

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
  optimize: () => void;
} & OptimizerOptions;

type LearningRate = number | ((iteration: number) => number);

class BaseOptimizer implements Optimizer {
  network: Network;
  learningRate: LearningRate;
  maxIterations: number;

  beforeIteration: OptimizerOptions["beforeIteration"];
  afterIteration: OptimizerOptions["afterIteration"];
  stopCondition: OptimizerOptions["stopCondition"];
  afterAll?: OptimizerOptions["afterAll"];

  doUpdate: (
    learningRate: number,
    data: OptimizationState,
    iteration: number
  ) => void = () => {};

  constructor(network: Network, options: OptimizerOptions) {
    this.learningRate = options.learningRate;
    this.maxIterations = options.maxIterations;
    this.network = network;

    this.beforeIteration = options.beforeIteration;
    this.afterIteration = options.afterIteration;
    this.stopCondition = options.stopCondition;
    this.afterAll = options.afterAll;
  }

  computeGradients = () => {
    const result = this.network.computeGradient();
    return {
      gradients: result.gradient,
      loss: result.loss,
    };
  };

  optimize: Optimizer["optimize"] = () => {
    for (let i = 0; i <= this.maxIterations; i += 1) {
      this.beforeIteration?.(this.network, i);

      const data = this.computeGradients();

      if (this.stopCondition?.(this.network, data, i)) {
        break;
      }

      const lr =
        typeof this.learningRate === "function"
          ? this.learningRate(i)
          : this.learningRate;

      if (lr < 0) {
        throw new Error("negative learningRate!");
      }

      this.doUpdate(lr, data, i);

      this.afterIteration?.(this.network, data, i);
    }

    this.afterAll?.();
  };
}

export class GradientDescentOptimizer extends BaseOptimizer {
  initialized = false;
  velocity: Matrix[] = [];
  momentum: number;

  currentWeights: Matrix[] = [];

  constructor(
    network: Network,
    options: OptimizerOptions & { momentum?: number; nesterov?: boolean }
  ) {
    super(network, options);

    this.momentum = options.momentum || 0;

    if (options.nesterov) {
      this.beforeIteration = (network, iteration) => {
        options.beforeIteration?.(network, iteration);
        this.currentWeights = network.layers.map((l) => l.weights);

        if (this.initialized) {
          network.layers.forEach((l, i) => {
            l.weights = l.weights.sum(this.velocity[i].scale(this.momentum));
          });
        }
      };
    }

    this.doUpdate = (learningRate, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
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

  private initialize(gradients: Matrix[]) {
    this.velocity = gradients.map((grad) => grad.scale(0));
    this.initialized = true;
  }
}

export class AdaGradOptimizer extends BaseOptimizer {
  initialized = false;
  gradientSquared: Matrix[] = [];

  lambda = Math.pow(10, -7);

  constructor(network: Network, options: OptimizerOptions) {
    super(network, options);

    this.doUpdate = (learningRate, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.gradientSquared = this.gradientSquared.map((grad, i) => {
        return grad.sum(gradients[i].hadamard(gradients[i]));
      });

      network.layers.forEach((layer, l) => {
        const weights = layer.weights;
        const weightDelta = this.gradientSquared[l].map((item) => {
          return learningRate / (this.lambda + Math.sqrt(item));
        });

        layer.updateWeights(
          weights.subtract(weightDelta.hadamard(gradients[l]))
        );
      });
    };
  }

  private initialize(gradients: Matrix[]) {
    this.gradientSquared = gradients.map((grad) => grad.scale(0));
    this.initialized = true;
  }
}

export class AdaDeltaOptimizer extends BaseOptimizer {
  initialized = false;

  gradientAverage: Matrix[] = [];

  deltaAverage: Matrix[] = [];

  lambda = Math.pow(10, -7);

  decay: number;

  constructor(network: Network, options: OptimizerOptions & { decay: number }) {
    super(network, options);

    this.decay = options.decay;

    this.doUpdate = (_, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.gradientAverage = this.gradientAverage.map((grad, l) => {
        return this.calcMovingAverage(grad, gradients[l]);
      });

      network.layers.forEach((layer, l) => {
        const weights = layer.weights;
        const weightDelta = this.deltaAverage[l].map((item, i, j) => {
          const deltaRMS = Math.sqrt(item + this.lambda);
          const gradRMS = Math.sqrt(
            this.gradientAverage[l].get(i, j) + this.lambda
          );
          return -(deltaRMS / gradRMS) * gradients[l].get(i, j);
        });

        this.deltaAverage[l] = this.calcMovingAverage(
          this.deltaAverage[l],
          weightDelta
        );

        layer.updateWeights(weights.sum(weightDelta));
      });
    };
  }

  private calcMovingAverage(previous: Matrix, vect: Matrix) {
    return previous
      .scale(this.decay)
      .sum(vect.hadamard(vect).scale(1 - this.decay));
  }

  private initialize(gradients: Matrix[]) {
    this.gradientAverage = gradients.map((grad) => grad.scale(0));
    this.deltaAverage = gradients.map((grad) => grad.scale(0));
    this.initialized = true;
  }
}

export class RMSPropOptimizer extends BaseOptimizer {
  initialized = false;

  gradientAverage: Matrix[] = [];

  lambda = Math.pow(10, -7);

  decay: number;

  constructor(network: Network, options: OptimizerOptions & { decay: number }) {
    super(network, options);

    this.decay = options.decay;

    this.doUpdate = (learningRate, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.gradientAverage = this.gradientAverage.map((grad, l) => {
        return this.calcMovingAverage(grad, gradients[l]);
      });

      network.layers.forEach((layer, l) => {
        const weights = layer.weights;
        const weightDelta = this.gradientAverage[l].map((item, i, j) => {
          const gradRMS = Math.sqrt(item + this.lambda);
          return -(learningRate / gradRMS) * gradients[l].get(i, j);
        });

        layer.updateWeights(weights.sum(weightDelta));
      });
    };
  }

  private calcMovingAverage(previous: Matrix, vect: Matrix) {
    return previous
      .scale(this.decay)
      .sum(vect.hadamard(vect).scale(1 - this.decay));
  }

  private initialize(gradients: Matrix[]) {
    this.gradientAverage = gradients.map((grad) => grad.scale(0));
    this.initialized = true;
  }
}

export class AdamOptimizer extends BaseOptimizer {
  initialized = false;

  gradientAverage: Matrix[] = [];
  momentAverage: Matrix[] = [];

  lambda = Math.pow(10, -8);

  gradientDecay: number; // beta_1
  momentDecay: number; // beta_2

  constructor(
    network: Network,
    options: OptimizerOptions & { gradientDecay: number; momentDecay: number }
  ) {
    super(network, options);

    this.gradientDecay = options.gradientDecay;
    this.momentDecay = options.momentDecay;

    this.doUpdate = (learningRate, data, iteration) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.momentAverage = this.momentAverage.map((mom, l) => {
        return mom
          .scale(this.momentDecay)
          .sum(gradients[l].scale(1 - this.momentDecay));
      });

      this.gradientAverage = this.gradientAverage.map((grad, l) => {
        return grad
          .scale(this.gradientDecay)
          .sum(
            gradients[l].hadamard(gradients[l]).scale(1 - this.gradientDecay)
          );
      });

      const correctedMoment = this.momentAverage.map((mom) => {
        return mom.scale(1 / (1 - Math.pow(this.momentDecay, iteration + 1)));
      });

      const correctedGradient = this.gradientAverage.map((grad) => {
        return grad.scale(
          1 / (1 - Math.pow(this.gradientDecay, iteration + 1))
        );
      });

      network.layers.forEach((layer, l) => {
        const weights = layer.weights;

        const weightDelta = correctedGradient[l].map((item, i, j) => {
          const gradRMS = Math.sqrt(item) + this.lambda;
          return -(learningRate / gradRMS) * correctedMoment[l].get(i, j);
        });

        layer.updateWeights(weights.sum(weightDelta));
      });
    };
  }

  private initialize(gradients: Matrix[]) {
    this.gradientAverage = gradients.map((grad) => grad.scale(0));
    this.momentAverage = gradients.map((grad) => grad.scale(0));
    this.initialized = true;
  }
}
