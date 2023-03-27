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
  return matrix.map((row) => row.filter((_, i) => i !== col));
};

export const transpose = (matrix: Matrix): Matrix => {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
};

export const multiply = (matrixA: Matrix, matrixB: Matrix): Matrix => {
  const result = new Array(matrixA.length)
    .fill(0)
    .map(() => new Array(matrixB[0].length).fill(0));

  return result.map((row, i) => {
    return row.map((_, j) => {
      return matrixA[i].reduce((sum, elm, k) => sum + elm * matrixB[k][j], 0);
    });
  });
};

/**
 * Sum of two matrixes.
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

export const clone = (matrix: Matrix): Matrix => {
  return mapMatrix(matrix, (x) => x);
};

export const mapMatrix = (
  matrix: Matrix,
  mapper: (value: number, i: number, j: number) => number
): Matrix => {
  return matrix.map((row, i) => row.map((x, j) => mapper(x, i, j)));
};
