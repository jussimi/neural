import { Matrix } from "./Matrix";

export type ErrorFN = {
  loss: (output: Matrix, expected: Matrix) => number;
  grad: (output: Matrix, expected: Matrix) => Matrix;
};

/**
 * Binary cross-entropy.
 */
export const LogLoss: ErrorFN = {
  loss: (output, expected) => {
    const t = output.get(0, 0);
    const y = expected.get(0, 0);
    return -(y * Math.log(t) + (1 - y) * Math.log(1 - t));
  },
  grad: (output, expected) => {
    const t = output.get(0, 0);
    const y = expected.get(0, 0);
    const result = (t - y) / (t - t * t);
    return Matrix.fromList([result]);
  },
};

/**
 * Cross-entropy.
 */
export const CELoss: ErrorFN = {
  loss: (output, expected) => {
    const result = output.map(Math.log).transpose().multiply(expected);
    return -result.get(0, 0);
  },
  grad: (output, expected) => {
    return output.map((t, i) => {
      const y = expected.get(i, 0);
      return -(y / t);
    });
  },
};

/**
 * Mean squared error.
 */
export const MSELoss: ErrorFN = {
  loss: (output, expected) => {
    let sum = 0;
    for (let i = 0; i < output.M; i += 1) {
      sum += Math.pow(output.get(i, 0) - expected.get(i, 0), 2);
    }
    return sum;
  },
  grad: (output, expected) => {
    return output.map((x, i, j) => {
      return x - expected.get(i, j);
    });
  },
};

export type ErrorFunctionKey = "cross-entropy" | "mean-squared" | "log-loss";
