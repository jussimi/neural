import { Matrix } from "./Matrix";

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
