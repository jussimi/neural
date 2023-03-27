import { ActivationFN } from "./activation";
import { Vector, Matrix } from "./Matrix";

export type LayerComputationResult = {
  /**
   * Neuron sum of weights*inputs
   */
  sum: Vector;
  /**
   * Sum passed through the activation function
   */
  activated: Vector;
  /**
   * Gradient of the activation function.
   */
  gradient: Vector;
};

export class Layer {
  isOutput: boolean = false;

  neuronCount: number;

  weights: Matrix = new Matrix([[]]);

  activation: ActivationFN;

  constructor(activation: ActivationFN, neuronCount: number) {
    this.neuronCount = neuronCount;
    this.activation = activation;
  }

  computeResult = (input: Vector): LayerComputationResult => {
    const sum = this.weights.multiply(input);
    const activated = this.activation.forward(sum, this.isOutput);
    const gradient = this.activation.backward(sum);

    return {
      sum,
      activated,
      gradient,
    };
  };

  initialize = (weights: Matrix, isOutput = false) => {
    this.weights = weights;
    this.isOutput = isOutput;
  };
}
