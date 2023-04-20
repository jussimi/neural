import { ActivationFunctionKey, Identity } from "./activation";
import { ErrorFunctionKey } from "./error";
import { ForwardPassResult, Layer } from "./Layer";
import { Matrix } from "./Matrix";
import { randomWeights } from "./utils";

export type DataSet = { input: number[]; output: number[] }[];
export class Network {
  error: ErrorFunctionKey;
  dataset: DataSet;
  layers: Layer[] = [];

  constructor(dataset: DataSet, error: ErrorFunctionKey) {
    this.dataset = dataset;
    this.error = error;
  }

  setData = (dataset: DataSet) => {
    this.dataset = dataset;
  };

  computeGradient = () => {
    // Initialize empty weight-matrices.
    const totalGradient = this.layers.map((layer) => layer.weights.scale(0));
    let totalLoss = 0;

    const results = this.dataset.map(({ input, output: expected }) => {
      const result = this.computeResult(
        Matrix.fromList(input),
        Matrix.fromList(expected)
      );

      // Update total gradient.
      const { activatedInput, results } = result;
      for (let l = 0; l < totalGradient.length; l += 1) {
        const weights = totalGradient[l];
        const activations = l === 0 ? activatedInput : results[l - 1].activated;
        const delta = result.deltas[l];
        for (let i = 0; i < weights.M; i += 1) {
          for (let j = 0; j < weights.N; j += 1) {
            const diff = delta.get(i, 0) * activations.get(j, 0);
            const newValue = weights.get(i, j) + diff / this.dataset.length;
            weights.set(i, j, newValue);
          }
        }
      }

      // Calculate total-loss.
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

  computeResult = (realInput: Matrix, expected: Matrix) => {
    // Calculates neuron-sums and activations for each layer.
    const data = this.forwardPass(realInput, expected);

    // Backropagate
    const deltas = this.backwardPass(expected, data.results);

    return {
      ...data,
      deltas,
    };
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

  backwardPass = (expected: Matrix, results: ForwardPassResult[]) => {
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

    return deltas;
  };

  validate = ({ input, output }: DataSet[1]) => {
    return this.forwardPass(Matrix.fromList(input), Matrix.fromList(output));
  };

  add = (activation: ActivationFunctionKey, neuronCount: number) => {
    this.layers.push(new Layer(activation, neuronCount, this.error));
  };

  initialize = (weights?: number[][][]) => {
    if (!this.layers.length) throw new Error("You need to specify layers");
    if (weights && weights.length !== this.layers.length) {
      throw new Error(
        `Invalid amount of weights supplied. Should be ${this.layers.length}.`
      );
    }

    // First dimension is based on the first input of the dataset.
    let currentDimension = this.dataset[0].input.length;
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
