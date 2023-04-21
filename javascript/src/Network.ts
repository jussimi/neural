import { ActivationFunctionKey, Identity } from "./Activation";
import { Dataset, DatasetItem } from "./Dataset";
import { ErrorFunctionKey } from "./Error";
import { ForwardPassResult, Layer } from "./Layer";
import { Matrix } from "./Matrix";
import { isCorrectCategory, randomWeights } from "./utils";

export type SetResult = {
  loss: number;
  percentage: number;
};

export class Network {
  error: ErrorFunctionKey;
  currentBatch: DatasetItem[];
  layers: Layer[] = [];

  trainSet: Dataset;
  testSet: Dataset | null;

  constructor(
    trainSet: Dataset,
    testSet: Dataset | null,
    error: ErrorFunctionKey
  ) {
    this.trainSet = trainSet;
    this.testSet = testSet;
    this.currentBatch = this.trainSet.generateBatch();
    this.error = error;
  }

  setBatch = (batch: DatasetItem[]) => {
    this.currentBatch = batch;
  };

  /**
   * Calculates forward and backward passes for each item in the currentBatch.
   *   - then calculates the total-loss and the gradient for weights.
   */
  computeGradient = () => {
    // Initialize empty gradient-matrices.
    const totalGradient = this.layers.map((layer) => layer.weights.scale(0));
    let totalLoss = 0;

    const batchLen = this.currentBatch.length;

    // Loop items in batch.
    const results = this.currentBatch.map(({ input, output: expected }) => {
      const data = this.computeResult(
        Matrix.fromList(input),
        Matrix.fromList(expected)
      );
      const { activatedInput, deltas, loss, results } = data;

      // Update total gradient.
      for (let layer = 0; layer < totalGradient.length; layer += 1) {
        const weights = totalGradient[layer];
        const activations =
          layer === 0 ? activatedInput : results[layer - 1].activated;
        const delta = deltas[layer];

        for (let i = 0; i < weights.M; i += 1) {
          for (let j = 0; j < weights.N; j += 1) {
            const diff = (delta.get(i, 0) * activations.get(j, 0)) / batchLen;
            const newValue = weights.get(i, j) + diff;
            weights.set(i, j, newValue);
          }
        }
      }

      // Update total-loss.
      totalLoss += loss / batchLen;

      return {
        input,
        output: expected,
        ...data,
      };
    });

    return {
      results,
      gradient: totalGradient,
      loss: totalLoss,
    };
  };

  /**
   * Calculates the backward and forward pass for a given input-output pair.
   */
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

    let deltas = [deltaOut];
    // Loop layers in reverse order starting from the second to last layer.
    //  - note that the next-delta is the first element in the deltas-array.
    for (let i = this.layers.length - 2; i >= 0; i -= 1) {
      const result = results[i];

      const delta = this.layers[i].backwardPass(
        result,
        deltas[0],
        this.layers[i + 1]
      );
      deltas.unshift(delta);
    }

    return deltas;
  };

  /**
   * Returns the forward-pass-result for an input-output pair.
   *  - contains estimate and the loss for the estimate.
   */
  predict({ input, output }: DatasetItem) {
    return this.forwardPass(Matrix.fromList(input), Matrix.fromList(output));
  }

  /**
   * Calculates the total-loss and prediction percentage for the whole dataset.
   */
  validateDataset(data: DatasetItem[]): SetResult {
    let totalLoss = 0;
    let correctCount = 0;

    const error = this.error;

    data.forEach((point) => {
      const { loss, estimate } = this.predict(point);
      totalLoss += loss;

      if (error === "log-loss") {
        const result = Math.round(estimate.get(0, 0));
        if (result === point.output[0]) {
          correctCount += 1;
        }
      } else if (error === "cross-entropy") {
        if (isCorrectCategory(estimate, point.output)) {
          correctCount += 1;
        }
      }
    });
    return {
      loss: totalLoss / data.length,
      percentage: correctCount / data.length,
    };
  }

  /**
   * Adds a new layer to the network.
   */
  add(activation: ActivationFunctionKey, neuronCount: number) {
    this.layers.push(new Layer(activation, neuronCount, this.error));
  }

  /**
   * Initializes weights in the network. The weights can also given directly.
   */
  initialize(weights?: number[][][]) {
    if (!this.layers.length) throw new Error("You need to specify layers");
    if (weights && weights.length !== this.layers.length) {
      throw new Error(
        `Invalid amount of weights supplied. Should be ${this.layers.length}.`
      );
    }

    // First dimension is based on the first input of the dataset.
    let currentDimension = this.currentBatch[0].input.length;
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
  }
}
