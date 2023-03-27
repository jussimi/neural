import { Identity } from "./activation";
import { ErrorFN } from "./error";
import { Layer, LayerComputationResult } from "./Layer";
import {
  hadamard,
  mapMatrix,
  Matrix,
  multiply,
  omit,
  scale,
  sum,
  transpose,
  Vector,
} from "./Matrix";
import { Optimizer } from "./Optimizer";

export type DataSet = [Vector, Vector][];
export class Network {
  error: ErrorFN;
  dataset: DataSet;
  optimizer: Optimizer;
  layers: Layer[];

  constructor(
    dataset: DataSet,
    error: ErrorFN,
    optimizer: Optimizer,
    layers: Layer[]
  ) {
    if (!layers.length) throw new Error("You need to specify layers");
    this.dataset = dataset;
    this.error = error;
    this.optimizer = optimizer;
    this.layers = layers;
  }

  setData = (dataset: DataSet) => {
    this.dataset = dataset;
  };

  optimize = () => {
    const computeGradients = () => {
      const result = this.computeGradient();
      return {
        gradients: result.gradient,
        loss: result.loss,
      };
    };
    this.optimizer.optimize(this, computeGradients);
  };

  computeGradient = () => {
    // Initialize empty weight-matrices.
    const totalGradient = this.layers.map((layer) => scale(layer.weights, 0));
    let totalLoss = 0;

    const results = this.dataset.map(([input, output]) => {
      const result = this.computeResult(input, output);

      // Calculate total-loss and total-gradient.
      const avg = 1 / this.dataset.length;
      totalLoss += result.loss * avg;
      result.gradients.forEach((grads, i) => {
        totalGradient[i] = sum(totalGradient[i], scale(grads, avg));
      });

      return {
        input,
        output,
        ...result,
      };
    });

    return {
      results,
      gradient: totalGradient,
      loss: totalLoss,
    };
  };

  computeResult = (realInput: Vector, output: Vector) => {
    // Adds constant 1 to input. Grows dimension by 1 to account for bias term in weight-matrice.
    const input = Identity.forward(realInput, false);

    // Calculates neuron-sums, activations and gradients of activations for each layer.
    const results = this.forwardPass(input);

    // Backropagate
    const gradients = this.backwardPass(input, output, results);

    const estimate = results[results.length - 1].activated;
    return {
      gradients,
      estimate: estimate,
      loss: this.error.loss(estimate, output),
    };
  };

  forwardPass = (input: Vector) => {
    // Keep track of results for each layer.
    const results: LayerComputationResult[] = [];

    let currentActivation = input;
    for (let i = 0; i < this.layers.length; i += 1) {
      const layer = this.layers[i];
      const result = layer.computeResult(currentActivation);
      currentActivation = result.activated;
      results.push(result);
    }

    return results;
  };

  backwardPass = (
    input: Vector,
    output: Vector,
    results: LayerComputationResult[]
  ) => {
    const resultOut = results[results.length - 1];

    const deltaOut = hadamard(
      this.error.grad(resultOut.activated, output),
      resultOut.gradient
    );

    let deltas = [deltaOut];
    for (let i = this.layers.length - 2; i >= 0; i -= 1) {
      const result = results[i];
      const weights = transpose(omit(this.layers[i + 1].weights, 0));
      const delta = hadamard(multiply(weights, deltas[0]), result.gradient); // delta^(l+1) = deltas[0]
      deltas.unshift(delta);
    }

    return this.layers.map((layer, l) => {
      const activations = l === 0 ? input : results[l - 1].activated;
      const delta = deltas[l];
      return mapMatrix(
        layer.weights,
        (_, i, j) => delta[i][0] * activations[j][0]
      );
    });
  };

  initialize = (weights?: Matrix[]) => {
    if (weights && weights.length !== this.layers.length) {
      throw new Error(
        `Invalid amount of weights supplied. Should be ${this.layers.length}.`
      );
    }

    // First dimension is based on the first input of the dataset.
    let currentDimension = this.dataset[0][0].length;
    this.layers.map((layer, l) => {
      const m = layer.neuronCount;
      const n = currentDimension + 1;

      currentDimension = layer.neuronCount;

      let w = weights?.[l];
      if (w) {
        if (w.length !== m || w[0].length !== n) {
          throw new Error(
            `Invalid weight dimension at layer ${l}. Should be ${m}x${n}.`
          );
        }
      } else {
        w = randomWeights(m, n);
      }
      layer.initialize(w, l === this.layers.length - 1);
    });
  };
}

function randomWeights(m: number, n: number) {
  const result: number[][] = [];
  for (let i = 0; i < m; i += 1) {
    let row: number[] = [];
    for (let i = 0; i < n; i += 1) {
      row.push(Math.random());
    }
    result.push(row);
  }
  return result;
}
