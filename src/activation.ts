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

export type ActivationFN = {
  forward: (sum: Matrix, isOutput: boolean) => Matrix;
  backward: (sum: Matrix, multiplyer: Matrix, isOutput: boolean) => Matrix;
};

export const ReLu: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarReLu.forward, sum, !isOutput),
  backward: (sum, multiplyer, isOutput) =>
    backpropagate(ScalarReLu.backward, sum, multiplyer, !isOutput),
};

export const TanH: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarTanH.forward, sum, !isOutput),
  backward: (sum, multiplyer, isOutput) =>
    backpropagate(ScalarTanH.backward, sum, multiplyer, !isOutput),
};

export const Sigmoid: ActivationFN = {
  forward: (sum, isOutput) => activate(ScalarSigmoid.forward, sum, !isOutput),
  backward: (sum, multiplyer, isOutput) =>
    backpropagate(ScalarSigmoid.backward, sum, multiplyer, !isOutput),
};

export const Softmax: ActivationFN = {
  forward: (input, isOutput) => {
    const result = SoftmaxFn(input);
    if (!isOutput) {
      return result.unshift(1);
    }

    return result;
  },
  backward: (sum, multiplyer, isOutput) => {
    const jacobianRows: number[][] = [];
    const sums = SoftmaxFn(sum);
    for (let i = 0; i < sum.M; i += 1) {
      jacobianRows[i] = [];
      const s_i = sums.get(i, 0);
      for (let j = 0; j < sum.M; j += 1) {
        if (i === j) {
          jacobianRows[i][j] = s_i * (1 - s_i);
        } else {
          jacobianRows[i][j] = -s_i * sums.get(j, 0);
        }
      }
    }
    const jacobian = Matrix.fromArrays(jacobianRows);
    const result = multiplyer.transpose().multiply(jacobian).transpose();
    return result;
  },
};

export const Identity: ActivationFN = {
  forward: (sum, isOutput) => activate((sum) => sum, sum, !isOutput),
  backward: (sum, _, isOutput) => activate((sum) => sum, sum, !isOutput),
};

function SigmoidFn(x: number) {
  const divisor = 1 + Math.exp(-x);
  return 1 / divisor;
}

function TanhFn(x: number) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
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
    sum += Math.exp(value - max);
  });

  return vect.map((x) => Math.exp(x - max) / sum);
}

function backpropagate(
  fn: ScalarActivationFN["backward"],
  sum: Matrix,
  multiplyer: Matrix,
  omitBias: boolean
) {
  let gradient = sum.map(fn);
  if (omitBias) {
    gradient = gradient.unshift(1);
  }
  const result = multiplyer.hadamard(gradient);
  return omitBias ? result.omit(0) : result;
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
