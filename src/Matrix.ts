export type Matrix = number[][];
export type Vector = Matrix;
export type Scalar = Matrix;

export const fromVect = (arr: number[]): Matrix => {
  return transpose([arr]);
};

export const getIndex = (matrix: Matrix, row = 0, column = 0): number => {
  return matrix[row][column];
};

export const omit = (matrix: Matrix, col: number): Matrix => {
  const result: Matrix = [];
  for (let i = 0; i < matrix.length; i += 1) {
    const row: number[] = [];
    for (let j = 0; j < matrix[0].length; j += 1) {
      if (j !== col) {
        row.push(matrix[i][j]);
      }
    }
    result.push(row);
  }
  return result;
};

export const transpose = (matrix: Matrix): Matrix => {
  const result: Matrix = [];
  for (let i = 0; i < matrix[0].length; i += 1) {
    result[i] = [];
    for (let j = 0; j < matrix.length; j += 1) {
      result[i][j] = matrix[j][i];
    }
  }
  return result;
};

export const multiply = (matrixA: Matrix, matrixB: Matrix): Matrix => {
  const result: Matrix = [];
  for (let i = 0; i < matrixA.length; i += 1) {
    result[i] = [];
    for (let j = 0; j < matrixB[0].length; j += 1) {
      let sum = 0;
      for (let k = 0; k < matrixA[0].length; k += 1) {
        sum += matrixA[i][k] * matrixB[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
};

/**
 * Sum of two matrices.
 */
export const sum = (matrixA: Matrix, matrixB: Matrix): Matrix => {
  return mapMatrix(matrixA, (x, i, j) => x + matrixB[i][j]);
};

/**
 * Hadamard product.
 */
export const hadamard = (matrixA: Matrix, matrixB: Matrix): Matrix => {
  return mapMatrix(matrixA, (x, i, j) => x * matrixB[i][j]);
};

/**
 * Matrix multiplied by scalar.
 */
export const scale = (matrix: Matrix, value: number): Matrix => {
  return mapMatrix(matrix, (x) => x * value);
};

export const mapMatrix = (
  matrix: Matrix,
  mapper: (value: number, i: number, j: number) => number
): Matrix => {
  const result: Matrix = [];
  for (let i = 0; i < matrix.length; i += 1) {
    result[i] = [];
    for (let j = 0; j < matrix[0].length; j += 1) {
      result[i][j] = mapper(matrix[i][j], i, j);
    }
  }
  return result;
};
