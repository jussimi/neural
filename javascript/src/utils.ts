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

export const isCorrectMultiCategory = (
  estimate: number[],
  output: number[]
) => {
  const label = output.indexOf(1);
  const values = estimate;
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

export function Smote(items: number[][], minLen: number, nearestK = 5) {
  const sampled: number[][] = [...items];

  const clonesPerPoint = Math.floor(minLen / items.length);

  type Neighbor = { distance: number; item: number[] };

  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const neighbors: Neighbor[] = [];
    for (let j = 0; j < items.length; j += 1) {
      if (i !== j) {
        const distance = euclidDistance(item, items[j]);
        neighbors.push({ distance, item: items[j] });
      }
    }
    const nearest = neighbors
      .sort((a, b) => a.distance - b.distance)
      .slice(0, nearestK);

    for (let k = 0; k < clonesPerPoint; k += 1) {
      const sample: number[] = [];
      for (let j = 0; j < items[i].length; j += 1) {
        const nextIndex = Math.floor(Math.random() * nearestK);
        const neighbor = nearest[nextIndex];
        const diff = neighbor.item[j] - item[j];
        sample.push(item[j] + diff * Math.random());
      }
      sampled.push(sample);
    }
  }

  return sampled.sort(() => Math.random() - 0.5);
}

export function euclidDistance(vect1: number[], vect2: number[]) {
  let sum = 0;

  for (let i = 0; i < vect1.length; i += 1) {
    sum += Math.pow(vect2[i] - vect1[i], 2);
  }
  return Math.sqrt(sum);
}
