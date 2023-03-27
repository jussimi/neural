import { Vector, Scalar } from "./Matrix";

export type ErrorFN = {
  loss: (output: Vector, expected: Vector) => number;
  grad: (output: Vector, expected: Vector) => Vector;
};

export const LogLoss: ErrorFN = {
  loss: (output, expected) => {
    const t = output.get();
    const y = expected.get();
    return -(y * Math.log(t) + (1 - y) * Math.log(1 - t));
  },
  grad: (output, expected) => {
    const t = output.get();
    const y = expected.get();
    const result = (t - y) / (t - t * t);
    return new Scalar(result);
  },
};

export const MSELoss: ErrorFN = {
  loss: (output, expected) => {
    return output.rows.reduce((sum, row, i) => {
      return sum + Math.pow(row[0] - expected.rows[i][0], 2);
    }, 0);
  },
  grad: (output, expected) => {
    return output.map((x, i, j) => {
      return x - expected.get(i, j);
    });
  },
};
