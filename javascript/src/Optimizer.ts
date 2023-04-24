import { Matrix } from "./Matrix";
import { Network, SetResult } from "./Network";

type OptimizationState = { loss: number; gradients: Matrix[] };
type OptimizerOptionFN<T> = (
  network: Network,
  payload: OptimizationState,
  iteration: number
) => T;

export type OptimizerOptions = {
  epochs: number;
  learningRate: LearningRate;
  beforeIteration?: (network: Network, iteration: number) => void;
  afterIteration?: OptimizerOptionFN<void>;
  afterEpoch?: (epoch: number) => void;
  stopCondition?: OptimizerOptionFN<boolean>;
  afterAll?: () => void;
};

export type Optimizer = {
  optimize: (shouldInitialize?: boolean) => {
    results: EpochResult[];
    took: number;
  };
} & OptimizerOptions;

type LearningRate = number | ((iteration: number) => number);

export type EpochResult = {
  train: SetResult;
  test: SetResult;
  took: number;
};

class BaseOptimizer implements Optimizer {
  network: Network;
  learningRate: LearningRate;
  epochs: number;

  beforeIteration: OptimizerOptions["beforeIteration"];
  afterIteration: OptimizerOptions["afterIteration"];
  stopCondition: OptimizerOptions["stopCondition"];
  afterAll?: OptimizerOptions["afterAll"];
  afterEpoch?: OptimizerOptions["afterEpoch"];

  updateWeights: (
    learningRate: number,
    data: OptimizationState,
    iteration: number
  ) => void = () => {};

  constructor(network: Network, options: OptimizerOptions) {
    this.learningRate = options.learningRate;
    this.epochs = options.epochs;
    this.network = network;

    this.beforeIteration = options.beforeIteration;
    this.afterIteration = options.afterIteration;
    this.stopCondition = options.stopCondition;
    this.afterAll = options.afterAll;
    this.afterEpoch = options.afterEpoch;
  }

  computeGradients = () => {
    const result = this.network.computeGradient();
    return {
      gradients: result.gradient,
      loss: result.loss,
    };
  };

  optimize: Optimizer["optimize"] = (shouldInitialize = true) => {
    if (shouldInitialize) {
      this.network.initialize();
    }

    const trainSet = this.network.trainSet;
    const testSet = this.network.testSet;
    const setSize = trainSet.items.length;
    const iterationsPerEpoch = Math.floor(setSize / trainSet.batchSize);

    const trainStart = Date.now();
    const results: EpochResult[] = [];

    const validateSets = (epochStart: number, initial = false) => {
      const trainRes = this.network.validateDataset(trainSet.items);
      const testRes = this.network.validateDataset(testSet?.items || []);
      const result: EpochResult = {
        train: trainRes,
        test: testRes,
        took: Date.now() - epochStart,
      };
      console.log(initial ? "Initial score:" : "");
      console.log(JSON.stringify(result));
      results.push(result);
    };

    validateSets(trainStart, true);

    let lastLog = Date.now();
    outer: for (let epoch = 0; epoch < this.epochs; epoch += 1) {
      const start = Date.now();
      for (let index = 0; index < iterationsPerEpoch; index += 1) {
        // Print progress every 100ms.
        if (
          Date.now() - lastLog > 100 ||
          index === 0 ||
          index === iterationsPerEpoch - 1
        ) {
          process.stdout.cursorTo(0);
          process.stdout.write(
            `Epoch ${epoch + 1}: ${index + 1} / ${iterationsPerEpoch} (${
              (Date.now() - start) / 1000
            }s)`
          );
          lastLog = Date.now();
        }

        const i = epoch * iterationsPerEpoch + index;

        this.beforeIteration?.(this.network, i);

        // STEP 1: Computes gradients. (forward and backward pass)
        const data = this.computeGradients();

        if (this.stopCondition?.(this.network, data, i)) {
          break outer;
        }

        const lr =
          typeof this.learningRate === "function"
            ? this.learningRate(i)
            : this.learningRate;

        if (lr < 0) {
          throw new Error("negative learningRate!");
        }

        // STEP 2: Calls the update function of the optimizer.
        this.updateWeights(lr, data, i);

        this.afterIteration?.(this.network, data, i);

        // STEP 3: Create a new batch of data.
        this.network.generateBatch();
      }
      validateSets(start);
      this.afterEpoch?.(epoch);
    }
    const took = (Date.now() - trainStart) / 1000;

    return { took, results };
  };
}

export class GradientDescentOptimizer extends BaseOptimizer {
  initialized = false;
  velocity: Matrix[] = [];
  momentum: number;

  nesterovWeights: Matrix[] = [];

  constructor(
    network: Network,
    options: OptimizerOptions & { momentum?: number; nesterov?: boolean }
  ) {
    super(network, options);

    this.momentum = options.momentum || 0;

    if (options.nesterov) {
      this.beforeIteration = (network, iteration) => {
        options.beforeIteration?.(network, iteration);
        this.nesterovWeights = network.layers.map((l) => l.weights);

        if (this.initialized) {
          network.layers.forEach((l, i) => {
            l.weights = l.weights.sum(this.velocity[i].scale(this.momentum));
          });
        }
      };
    }

    this.updateWeights = (learningRate, data) => {
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
          weights = this.nesterovWeights[i];
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

    this.updateWeights = (learningRate, data) => {
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

    this.updateWeights = (_, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.gradientAverage = this.gradientAverage.map((grad, l) => {
        return calcMovingAverage(grad, gradients[l], this.decay);
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

        this.deltaAverage[l] = calcMovingAverage(
          this.deltaAverage[l],
          weightDelta,
          this.decay
        );

        layer.updateWeights(weights.sum(weightDelta));
      });
    };
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

    this.updateWeights = (learningRate, data) => {
      const { gradients } = data;
      if (!this.initialized) {
        this.initialize(gradients);
      }

      this.gradientAverage = this.gradientAverage.map((grad, l) => {
        return calcMovingAverage(grad, gradients[l], this.decay);
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

    this.updateWeights = (learningRate, data, iteration) => {
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

const calcMovingAverage = (previous: Matrix, vect: Matrix, decay: number) => {
  return previous.scale(decay).sum(vect.hadamard(vect).scale(1 - decay));
};
