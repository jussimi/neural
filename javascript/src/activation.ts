import { Matrix } from "./Matrix";

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
  backward: (x) => SigmoidFn(x) * (1 - SigmoidFn(x)),
};

export type ActivationFunctionKey = "sigmoid" | "tanh" | "relu" | "softmax";

export type ActivationFN = {
  forward: (sum: Matrix, isOutput: boolean) => Matrix;
  backward: (sum: Matrix) => Matrix;
  output: (output: Matrix, expected: Matrix) => Matrix;
};

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
    return sum.subtract(expected);
  },
  backward: (sum) => {
    // TODO
    throw new Error("Use softmax only on output!");
  },
};

export const Identity: ActivationFN = {
  forward: (sum, isOutput) => activate((sum) => sum, sum, !isOutput),
  backward: (sum) => activate((sum) => sum, sum, false),
  output: (sum) => activate((sum) => sum, sum, false),
};

function backpropagate(fn: ScalarActivationFN["backward"], sum: Matrix) {
  return sum.map(fn);
}

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

function clampedExp(x: number) {
  const range = 500;
  const clamped = Math.min(Math.max(-range, x), range);

  return Math.exp(clamped);
}