import { Matrix } from "./Matrix";

export type ActivationFunctionKey = "sigmoid" | "tanh" | "relu" | "softmax";

export type ActivationFN = {
  forward: (sum: Matrix, isOutput: boolean) => Matrix;
  backward: (sum: Matrix) => Matrix;
  output: (output: Matrix, expected: Matrix) => Matrix;
};

export class ActivationFunction implements ActivationFN {
  type: ActivationFunctionKey;

  forward: (sum: Matrix, isOutput: boolean) => Matrix;
  backward: (sum: Matrix) => Matrix;
  output: (output: Matrix, expected: Matrix) => Matrix;

  constructor(type: ActivationFunctionKey) {
    let activation: ActivationFN;
    switch (type) {
      case "softmax":
        activation = Softmax;
        break;
      case "sigmoid":
        activation = Sigmoid;
        break;
      case "tanh":
        activation = TanH;
        break;
      case "relu":
        activation = ReLu;
        break;
      default:
        throw new Error("Invalid activation function");
    }
    this.type = type;
    this.backward = activation.backward;
    this.forward = activation.forward;
    this.output = activation.output;
  }
}

export const ReLu: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarReLu.forward, sum, !isOutput),
  backward: (sum) => backpropagate(ScalarReLu.backward, sum),
  output: (sum) => backpropagate(ScalarReLu.backward, sum),
};

export const TanH: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarTanH.forward, sum, !isOutput),
  backward: (sum) => backpropagate(ScalarTanH.backward, sum),
  output: (sum) => backpropagate(ScalarTanH.backward, sum),
};

export const Sigmoid: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarSigmoid.forward, sum, !isOutput),
  backward: (sum) => backpropagate(ScalarSigmoid.backward, sum),
  output: (sum) => backpropagate(ScalarSigmoid.backward, sum),
};

export const Softmax: ActivationFN = {
  forward: (input, isOutput) => {
    const result = SoftmaxFn(input);
    if (!isOutput) {
      return result.unshift(1);
    }

    return result;
  },
  output: (sum, expected) => {
    // Only works with cross-entropy.
    return sum.subtract(expected);
  },
  backward: (sum) => {
    // TODO
    throw new Error("Use softmax only on output!");
  },
};

/**
 * Used for growing the dimension of the input layer.
 */
export const Identity: ActivationFN = {
  forward: (sum, isOutput) => activate((sum) => sum, sum, !isOutput),
  backward: (sum) => backpropagate((sum) => sum, sum),
  output: (sum) => backpropagate((sum) => sum, sum),
};

/**
 * A function from R -> R.
 */
export type ScalarActivationFN = {
  forward: (x: number) => number;
  backward: (x: number) => number;
};

const ScalarReLu: ScalarActivationFN = {
  forward: (x) => (x > 0 ? x : 0),
  backward: (x) => (x > 0 ? 1 : 0),
};

const ScalarTanH: ScalarActivationFN = {
  forward: TanhFn,
  backward: (x) => 1 - Math.pow(TanhFn(x), 2),
};

const ScalarSigmoid: ScalarActivationFN = {
  forward: SigmoidFn,
  backward: (x) => {
    const s = SigmoidFn(x);
    return s * (1 - s);
  },
};

/**
 * Generalizes a function derivative from R -> R to R^N -> R^N
 */
function backpropagate(fn: ScalarActivationFN["backward"], sum: Matrix) {
  return sum.map(fn);
}

/**
 * Generalizes a function from R -> R to R^N -> R^N+1
 *  - adds a constant 1 to the result if not the output layer.
 */
function activate(
  fn: ScalarActivationFN["forward"],
  sum: Matrix,
  appendBias: boolean
) {
  const vect = sum.map(fn);
  if (appendBias) {
    return vect.unshift(1);
  }
  return vect;
}

function SigmoidFn(x: number) {
  const divisor = 1 + clampedExp(-x);
  return 1 / divisor;
}

function TanhFn(x: number) {
  return (clampedExp(x) - clampedExp(-x)) / (clampedExp(x) + clampedExp(-x));
}

function SoftmaxFn(vect: Matrix) {
  let max = vect.get(0, 0);
  vect.iterate((value) => {
    if (value > max) {
      max = value;
    }
  });

  let sum = 0;
  vect.iterate((value) => {
    sum += clampedExp(value - max);
  });

  const res = vect.map((x) => clampedExp(x - max) / sum);
  return res;
}

// Ensures that the result is not 0 or infinity.
function clampedExp(x: number) {
  const range = 500;
  const clamped = Math.min(Math.max(-range, x), range);

  return Math.exp(clamped);
}
