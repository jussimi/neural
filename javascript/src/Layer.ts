import {
  ActivationFN,
  ActivationFunctionKey,
  ReLu,
  Sigmoid,
  Softmax,
  TanH,
} from "./activation";
import { CELoss, ErrorFN, ErrorFunctionKey, LogLoss, MSELoss } from "./error";
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

  activationKey: ActivationFunctionKey;

  activation: ActivationFN;

  private errorKey: ErrorFunctionKey = "cross-entropy";

  private error: ErrorFN = CELoss;

  constructor(
    activation: ActivationFunctionKey,
    neuronCount: number,
    error: ErrorFunctionKey
  ) {
    this.neuronCount = neuronCount;
    this.activationKey = activation;
    this.errorKey = error;

    switch (activation) {
      case "softmax":
        this.activation = Softmax;
        break;
      case "sigmoid":
        this.activation = Sigmoid;
        break;
      case "tanh":
        this.activation = TanH;
        break;
      case "relu":
        this.activation = ReLu;
        break;
      default:
        throw new Error("Invalid activation function");
    }

    switch (error) {
      case "cross-entropy":
        this.error = CELoss;
        break;
      case "log-loss":
        this.error = LogLoss;
        break;
      case "mean-squared":
        this.error = MSELoss;
        break;
      default:
        throw new Error("Invalid error function");
    }
  }

  forwardPass = (input: Matrix, expected: Matrix): ForwardPassResult => {
    const sum = this.weights.multiply(input);
    const activated = this.activation.forward(sum, this.isOutput);

    return {
      sum,
      activated,
      error: this.isOutput ? this.error.loss(activated, expected) : -1,
    };
  };

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

  outputPass = (result: ForwardPassResult, expected: Matrix) => {
    if (this.activationKey === "softmax") {
      if (this.errorKey !== "cross-entropy") {
        throw new Error("Can only use softmax with cross-entropy");
      }
      return this.activation.output(result.activated, expected);
    }

    return this.error
      .grad(result.activated, expected)
      .hadamard(this.activation.backward(result.sum), true);
  };

  updateWeights = (weights: Matrix) => {
    this.weights = weights;
    this.weightsTranspose = weights.omit(0).transpose();
  };

  initialize = (weights: Matrix, isOutput = false) => {
    this.isOutput = isOutput;
    this.updateWeights(weights);
  };
}