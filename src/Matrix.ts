export class Matrix {
  rows: number[][];

  constructor(rows: number[][]) {
    this.rows = rows;
  }

  get = (row = 0, column = 0) => {
    return this.rows[row][column];
  };

  omit = (col: number): Matrix => {
    const result = this.rows.map((row) => row.filter((_, i) => i !== col));
    return new Matrix(result);
  };

  transpose = (): Matrix => {
    const rows = this.rows[0].map((_, colIndex) =>
      this.rows.map((row) => row[colIndex])
    );
    return new Matrix(rows);
  };

  multiply = (matrix: Matrix): Matrix => {
    const result = new Array(this.rows.length)
      .fill(0)
      .map(() => new Array(matrix.rows[0].length).fill(0));

    const rows = result.map((row, i) => {
      return row.map((_, j) => {
        return this.rows[i].reduce(
          (sum, elm, k) => sum + elm * matrix.rows[k][j],
          0
        );
      });
    });
    return new Matrix(rows);
  };

  /**
   * Sum of two matrixes.
   */
  sum = (matrix: Matrix): Matrix => {
    return this.map((x, i, j) => x + matrix.rows[i][j]);
  };

  /**
   * Hadamard product.
   */
  hadamard = (matrix: Matrix): Matrix => {
    return this.map((x, i, j) => x * matrix.rows[i][j]);
  };

  /**
   * Matrix multiplied by scalar.
   */
  scale = (value: number): Matrix => {
    return this.map((x) => x * value);
  };

  map = (mapper: (value: number, i: number, j: number) => number): Matrix => {
    const rows = this.rows.map((row, i) => row.map((x, j) => mapper(x, i, j)));
    return new Matrix(rows);
  };

  clone = (): Matrix => {
    return this.map((x) => x);
  };
}

/**
 * Util to create a vertical vector from list of numbers.
 */
export class Vector extends Matrix {
  constructor(row: number[]) {
    super([row]);
    this.rows = this.transpose().rows;
  }
}

/**
 * Creates a matrix from a scalar-value.
 */
export class Scalar extends Matrix {
  constructor(value: number) {
    super([[value]]);
  }
}
