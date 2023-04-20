import { Matrix } from "./Matrix";
import { DataSet } from "./Network";

/**
 * Returns one-hot encoded list of specified length.
 *  - 1 is at given position.
 */
export const oneHotEncode = (label: number, len: number): number[] => {
  const vector = [...Array(len).keys()].map(() => 0);
  vector[label] = 1;
  return vector;
};

export const isCorrectCategory = (estimate: Matrix, output: number[]) => {
  const label = output.indexOf(1);
  const values = estimate.items;
  const max = Math.max(...values);
  return label === values.indexOf(max);
};

export const generateBatch = (dataset: DataSet, batchSize: number): DataSet => {
  const set: DataSet = [];
  const usedIndices = new Map<number, number>();

  while (set.length < batchSize) {
    inner: while (true) {
      const nextIndex = Math.floor(Math.random() * dataset.length);
      if (usedIndices.has(nextIndex)) {
        continue;
      }
      usedIndices.set(nextIndex, 1);
      const data = dataset[nextIndex];
      set.push(data);
      break inner;
    }
  }
  return set;
};

export function randomWeights(m: number, n: number) {
  const result: number[][] = [];
  for (let i = 0; i < m; i += 1) {
    let row: number[] = [];
    row[0] = 0;
    for (let j = 1; j < n; j += 1) {
      row.push(getRandomBetween(-0.5, 0.5));
    }
    result.push(row);
  }
  return result;
}

export function getRandomBetween(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

export type ValidationRes = {
  loss: number;
  percentage: number;
};
export type Result = {
  iteration: number;
  train: ValidationRes;
  test: ValidationRes;
};
