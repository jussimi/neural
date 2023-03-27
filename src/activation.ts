import { Vector } from "./Matrix";

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

export type ActivationFN = {
  forward: (x: Vector, isOutput: boolean) => Vector;
  backward: (x: Vector) => Vector;
};

export const ReLu: ActivationFN = {
  forward: (x, isOutput) => activate(ScalarReLu.forward, x, !isOutput),
  backward: (x) => activate(ScalarReLu.backward, x),
};

export const TanH: ActivationFN = {
  forward: (x, isOutput) => activate(ScalarTanH.forward, x, !isOutput),
  backward: (x) => activate(ScalarTanH.backward, x),
};

export const Sigmoid: ActivationFN = {
  forward: (x, isOutput) => activate(ScalarSigmoid.forward, x, !isOutput),
  backward: (x) => activate(ScalarSigmoid.backward, x),
};

export const Softmax: ActivationFN = {
  forward: (input, isOutput) => {
    const sum = input.rows.reduce((s, x) => s + Math.exp(x[0]), 0);
    const result = input.map((x) => x / sum);
    if (!isOutput) {
      result.rows.unshift([1]);
    }
    return result;
  },
  backward: (x) => {
    return x;
  },
};

export const Identity: ActivationFN = {
  forward: (x, isOutput) => activate((x) => x, x, !isOutput),
  backward: (x) => activate((x) => x, x),
};

function SigmoidFn(x: number) {
  const divisor = 1 + Math.exp(-x);
  return 1 / divisor;
}

function TanhFn(x: number) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

function activate(
  fn: ScalarActivationFN["forward"],
  input: Vector,
  appendBias = false
) {
  const vect = input.map((x) => fn(x));
  if (appendBias) {
    vect.rows.unshift([1]);
  }
  return vect;
}
