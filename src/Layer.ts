import { ActivationFN } from "./activation";
import { Matrix } from "./Matrix";

export type LayerComputationResult = {
  /**
   * Neuron sum of weights*inputs
   */
  sum: Matrix;
  /**
   * Sum passed through the activation function
   */
  activated: Matrix;
  /**
   * Gradient of the activation function.
   */
  gradient: Matrix;
};

export type ForwardPassResult = {
  /**
   * Neuron sum of weights*inputs
   */
  sum: Matrix;
  /**
   * Sum passed through the activation function
   */
  activated: Matrix;
};

export type BackwardPassResult = {
  delta: Matrix;
};

export class Layer {
  isOutput: boolean = false;

  neuronCount: number;

  weights: Matrix = Matrix.fromList([]);

  activation: ActivationFN;

  constructor(activation: ActivationFN, neuronCount: number) {
    this.neuronCount = neuronCount;
    this.activation = activation;
  }

  forwardPass = (input: Matrix): ForwardPassResult => {
    const sum = this.weights.multiply(input);
    const activated = this.activation.forward(sum, this.isOutput);

    return {
      sum,
      activated,
    };
  };

  /**
   * Multiplyer is a vector that should be multiplied with the jacobian.
   *  - On output layer it is grad(E_a)^T === transpose of the Error with respect to activation.
   *  - On other layers it is delta_next * W_next
   */
  backwardPass = (result: ForwardPassResult, multiplyer: Matrix): Matrix => {
    const delta = this.activation.backward(
      result.sum,
      multiplyer,
      this.isOutput
    );

    return delta;
  };

  initialize = (weights: Matrix, isOutput = false) => {
    this.weights = weights;
    this.isOutput = isOutput;
  };
}
