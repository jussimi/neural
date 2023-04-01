import { ActivationFunctionKey, Identity } from "./activation";
import { ErrorFunctionKey } from "./error";
import { ForwardPassResult, Layer } from "./Layer";
import { Matrix } from "./Matrix";
import { Optimizer } from "./Optimizer";

export type DataSet = [number[], number[]][];
export class Network {
  error: ErrorFunctionKey;
  dataset: DataSet;
  optimizer: Optimizer;
  layers: Layer[] = [];

  constructor(dataset: DataSet, error: ErrorFunctionKey, optimizer: Optimizer) {
    this.dataset = dataset;
    this.error = error;
    this.optimizer = optimizer;
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
    const totalGradient = this.layers.map((layer) => layer.weights.scale(0));
    let totalLoss = 0;

    const results = this.dataset.map(([input, expected]) => {
      const result = this.computeResult(
        Matrix.fromList(input),
        Matrix.fromList(expected),
        totalGradient
      );

      // Calculate total-loss and total-gradient.
      totalLoss += result.loss / this.dataset.length;

      return {
        input,
        output: expected,
        ...result,
      };
    });

    return {
      results,
      gradient: totalGradient,
      loss: totalLoss,
    };
  };

  computeResult = (realInput: Matrix, expected: Matrix, total: Matrix[]) => {
    // Calculates neuron-sums and activations for each layer.
    const data = this.forwardPass(realInput, expected);

    // Backropagate
    this.backwardPass(data.activatedInput, expected, data.results, total);
    return data;
  };

  forwardPass = (input: Matrix, expected: Matrix) => {
    // Adds constant 1 to input. Grows dimension by 1 to account for bias term in weight-matrice.
    const activatedInput = Identity.forward(input, false);

    // Keep track of results for each layer.
    const results: ForwardPassResult[] = [];

    let currentActivation = activatedInput;
    for (let i = 0; i < this.layers.length; i += 1) {
      const layer = this.layers[i];
      const result = layer.forwardPass(currentActivation, expected);
      currentActivation = result.activated;
      results.push(result);
    }

    const { activated, error } = results[results.length - 1];
    return {
      loss: error,
      results,
      estimate: activated,
      activatedInput,
    };
  };

  backwardPass = (
    input: Matrix,
    expected: Matrix,
    results: ForwardPassResult[],
    total: Matrix[]
  ) => {
    const resultOut = results[results.length - 1];

    const layerOut = this.layers[this.layers.length - 1];

    const deltaOut = layerOut.outputPass(resultOut, expected);

    // Store transposes of deltas -> avoids transposing the weight matrix.
    let deltas = [deltaOut];
    for (let i = this.layers.length - 2; i >= 0; i -= 1) {
      const result = results[i];

      // delta^(l+1) = deltas[0]
      const delta = this.layers[i].backwardPass(
        result,
        deltas[0],
        this.layers[i + 1]
      );
      deltas.unshift(delta);
    }

    for (let l = 0; l < total.length; l += 1) {
      const weights = total[l];
      const activations = l === 0 ? input : results[l - 1].activated;
      const delta = deltas[l];
      for (let i = 0; i < weights.M; i += 1) {
        for (let j = 0; j < weights.N; j += 1) {
          const diff = delta.get(i, 0) * activations.get(j, 0);
          const newValue = weights.get(i, j) + diff / this.dataset.length;
          weights.set(i, j, newValue);
        }
      }
    }
  };

  validate = ([input, expected]: DataSet[1]) => {
    return this.forwardPass(Matrix.fromList(input), Matrix.fromList(expected));
  };

  add = (activation: ActivationFunctionKey, neuronCount: number) => {
    const layers = [
      ...this.layers,
      new Layer(activation, neuronCount, this.error),
    ];
    const clone = new Network(this.dataset, this.error, this.optimizer);
    clone.layers = layers;
    return clone;
  };

  train = () => {
    this.initialize();
    this.optimize();
  };

  initialize = (weights?: number[][][]) => {
    if (!this.layers.length) throw new Error("You need to specify layers");
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
      layer.initialize(Matrix.fromArrays(w), l === this.layers.length - 1);
    });
  };
}

function randomWeights(m: number, n: number) {
  const result: number[][] = [];
  for (let i = 0; i < m; i += 1) {
    let row: number[] = [];
    for (let i = 0; i < n; i += 1) {
      row.push(getRandomBetween(-0.5, 0.5));
    }
    result.push(row);
  }
  return result;
}

function getRandomBetween(min: number, max: number) {
  return Math.random() * (max - min) + min;
}
