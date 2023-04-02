/**
 * A class representing a Matrix.
 *  - Implemented using a flat Float64Array, since this is more efficient than normal array of arrays.
 *  - Loops are done using plain old for-loops, since they seem to be more efficient than functional methods.
 */
export class Matrix {
  M: number;
  N: number;

  private items: Float64Array;

  constructor(items: Float64Array, M: number, N: number) {
    this.M = M;
    this.N = N;
    this.items = items;
  }

  /**
   * Initialize matrix from an array of arrays.
   */
  static fromArrays(items: number[][]) {
    const m = items.length;
    const n = items[0].length;

    const list = new Float64Array(m * n);
    for (let i = 0; i < m; i += 1) {
      for (let j = 0; j < n; j += 1) {
        list[i * n + j] = items[i][j];
      }
    }
    return new Matrix(list, m, n);
  }

  /**
   * Initialize a Matrix from a single list.
   *  - used for creating vertical or horizontal vectors.
   */
  static fromList(items: number[], isVertical = true) {
    const m = isVertical ? items.length : 1;
    const n = isVertical ? 1 : items.length;

    const list = new Float64Array(items.length);
    for (let i = 0; i < items.length; i += 1) {
      list[i] = items[i];
    }

    return new Matrix(list, m, n);
  }

  get(i: number, j: number) {
    return this.items[calcIndex(i, j, this.N)];
  }

  set(i: number, j: number, value: number) {
    this.items[calcIndex(i, j, this.N)] = value;
  }

  /**
   * Returns matrix as normal 2d-list. Use for logging/debugging.
   */
  toArrays() {
    const result: number[][] = [];
    for (let i = 0; i < this.M; i += 1) {
      result[i] = [];
      for (let j = 0; j < this.N; j += 1) {
        result[i][j] = this.get(i, j);
      }
    }
    return result;
  }

  multiply(matrix: Matrix): Matrix {
    if (this.N !== matrix.M)
      throw new Error(
        `Dimension mismatch. Got ${this.M}x${this.N} and ${matrix.M}x${matrix.N}`
      );
    const result = new Float64Array(this.M * matrix.N);
    for (let i = 0; i < this.M; i++) {
      for (let j = 0; j < matrix.N; j++) {
        let sum = 0;
        for (let k = 0; k < this.N; k++) {
          sum += this.get(i, k) * matrix.get(k, j);
        }
        result[calcIndex(i, j, matrix.N)] = sum;
      }
    }
    return new Matrix(result, this.M, matrix.N);
  }

  transpose() {
    const result = new Float64Array(this.M * this.N);
    for (let i = 0; i < this.M; i += 1) {
      for (let j = 0; j < this.N; j += 1) {
        result[calcIndex(j, i, this.M)] = this.get(i, j);
      }
    }
    return new Matrix(result, this.N, this.M);
  }

  /**
   * Omits specified column from the matrix.
   */
  omit(col: number) {
    const result = new Float64Array(this.M * this.N - 1);

    let k = 0;
    for (let i = 0; i < this.M; i += 1) {
      for (let j = 0; j < this.N; j += 1) {
        if (col !== j) {
          result[k] = this.get(i, j);
          k += 1;
        }
      }
    }
    return new Matrix(result, this.M, this.N - 1);
  }

  sum(matrix: Matrix, inPlace = false) {
    return this.map((x, i, j) => x + matrix.get(i, j), inPlace);
  }

  subtract(matrix: Matrix, inPlace = false) {
    return this.map((x, i, j) => x - matrix.get(i, j), inPlace);
  }

  hadamard(matrix: Matrix, inPlace = false) {
    return this.map((x, i, j) => x * matrix.get(i, j), inPlace);
  }

  scale(value: number, inPlace = false) {
    return this.map((x) => x * value, inPlace);
  }

  map(mapper: (item: number, i: number, j: number) => number, inPlace = false) {
    const result = inPlace ? this.items : new Float64Array(this.M * this.N);
    for (let i = 0; i < this.M; i += 1) {
      for (let j = 0; j < this.N; j += 1) {
        result[calcIndex(i, j, this.N)] = mapper(this.get(i, j), i, j);
      }
    }
    return inPlace ? this : new Matrix(result, this.M, this.N);
  }

  // TODO: Could probably create a Vector-subclass.

  /**
   * Appends element to start of a vertical vector.
   */
  unshift(value: number) {
    if (this.N !== 1) throw new Error("Can only unshift vertical vectors!");
    const result = new Float64Array(this.M * this.N + 1);
    result[0] = value;
    for (let i = 0; i < this.M * this.N; i += 1) {
      result[i + 1] = this.items[i];
    }
    return new Matrix(result, this.M + 1, this.N);
  }

  /**
   * Iterate over a vertical vector.
   */
  iterate(iterator: (value: number, i: number) => void) {
    for (let i = 0; i < this.M; i += 1) {
      iterator(this.get(i, 0), i);
    }
  }
}

function calcIndex(i: number, j: number, N: number) {
  return i * N + j;
}
