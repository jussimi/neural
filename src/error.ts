import { Vector, mapMatrix } from "./Matrix";

export type ErrorFN = {
  loss: (output: Vector, expected: Vector) => number;
  grad: (output: Vector, expected: Vector) => Vector;
};

export const LogLoss: ErrorFN = {
  loss: (output, expected) => {
    const t = output[0][0];
    const y = expected[0][0];
    return -(y * Math.log(t) + (1 - y) * Math.log(1 - t));
  },
  grad: (output, expected) => {
    const t = output[0][0];
    const y = expected[0][0];
    const result = (t - y) / (t - t * t);
    return [[result]];
  },
};

export const MSELoss: ErrorFN = {
  loss: (output, expected) => {
    return output.reduce((sum, row, i) => {
      return sum + Math.pow(row[0] - expected[i][0], 2);
    }, 0);
  },
  grad: (output, expected) => {
    return mapMatrix(output, (x, i, j) => {
      return x - expected[i][j];
    });
  },
};
