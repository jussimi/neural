import { ActivationFunction, ActivationFunctionKey } from "./Activation";
import { ErrorFunction, ErrorFunctionKey } from "./Error";
import { Matrix } from "./Matrix";

export type ForwardPassResult = {
  /**
   * Neuron sum of weights*inputs
   */
  sum: Matrix;
  /**
   * Sum passed through the activation function
   */
  activated: Matrix;
  /**
   * Error of the result.
   */
  error: number;
};

export class Layer {
  isOutput: boolean = false;

  neuronCount: number;

  weights: Matrix = Matrix.fromList([]);

  // Transposed weights array, where bias terms have been omitted.
  weightsTranspose: Matrix = Matrix.fromList([]);

  activation: ActivationFunction;

  error: ErrorFunction;

  constructor(
    activation: ActivationFunctionKey,
    neuronCount: number,
    error: ErrorFunctionKey
  ) {
    this.neuronCount = neuronCount;

    this.activation = new ActivationFunction(activation);
    this.error = new ErrorFunction(error);
  }

  /**
   * Calculates the neuron sum, activation and the error (on output layer) for a given input-output pair.
   */
  forwardPass = (input: Matrix, expected: Matrix): ForwardPassResult => {
    const sum = this.weights.multiply(input);
    const activated = this.activation.forward(sum, this.isOutput);

    return {
      sum,
      activated,
      error: this.isOutput ? this.error.loss(activated, expected) : -1,
    };
  };

  /**
   * Calculate the backwardpass for the given forward-pass-result and the next weights and deltas.
   */
  backwardPass = (
    result: ForwardPassResult,
    deltaNext: Matrix,
    layerNext: Layer
  ): Matrix => {
    const weights = layerNext.weightsTranspose;

    return weights
      .multiply(deltaNext)
      .hadamard(this.activation.backward(result.sum), true);
  };

  /**
   * Calculate the output delta. Can be extended to use hard-coded activation/error pairs for a bit of a performance boost.
   *  - Example of cross-entropy with softmax activation.
   */
  outputPass = (result: ForwardPassResult, expected: Matrix) => {
    if (this.activation.type === "softmax") {
      if (this.error.type !== "cross-entropy") {
        throw new Error("Can only use softmax with cross-entropy");
      }
      return this.activation.output(result.activated, expected);
    }

    return this.error
      .grad(result.activated, expected)
      .hadamard(this.activation.backward(result.sum), true);
  };

  /**
   * Set the weights of the layer. Also computes the transpose for later use.
   */
  updateWeights = (weights: Matrix) => {
    this.weights = weights;
    this.weightsTranspose = weights.omit(0).transpose();
  };

  /**
   * Initializes the layer.
   */
  initialize = (weights: Matrix, isOutput = false) => {
    this.isOutput = isOutput;
    this.updateWeights(weights);
  };
}
