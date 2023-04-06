use std::convert::From;

#[derive(Debug)]
pub struct Matrix {
    pub items: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.items[calc_index(i, j, self.cols)]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.items[calc_index(i, j, self.cols)] = value
    }

    pub fn multiply(&self, matrix: &Matrix) -> Matrix {
        if self.cols != matrix.rows {
            panic!(
                "`Dimension mismatch. Got {}x{} and {}x{}",
                self.rows, self.cols, matrix.rows, matrix.cols
            )
        }
        let mut result: Vec<f64> = vec![0_f64; self.rows * matrix.cols];

        for i in 0..self.rows {
            for j in 0..matrix.cols {
                let mut sum = 0_f64;
                for k in 0..self.cols {
                    sum += self.get(i, k) * matrix.get(k, j);
                }
                result[calc_index(i, j, matrix.cols)] = sum;
            }
        }

        create_matrix(result, self.rows, matrix.cols)
    }

    pub fn transpose(&self) -> Matrix {
        let mut result: Vec<f64> = vec![0_f64; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[calc_index(j, i, self.rows)] = self.get(i, j)
            }
        }
        create_matrix(result, self.cols, self.rows)
    }

    pub fn omit(&self, col: usize) -> Matrix {
        let mut result: Vec<f64> = vec![0_f64; self.rows * (self.cols - 1)];
        let mut k = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if col != j {
                    result[k] = self.get(i, j);
                    k += 1
                }
            }
        }
        create_matrix(result, self.rows, self.cols - 1)
    }

    pub fn sum(&self, matrix: &Matrix) -> Matrix {
        return self.map(&|item: f64, i: usize, j: usize| -> f64 { item + matrix.get(i, j) });
    }

    pub fn subtract(&self, matrix: &Matrix) -> Matrix {
        return self.map(&|item: f64, i: usize, j: usize| -> f64 { item - matrix.get(i, j) });
    }

    pub fn hadamard(&self, matrix: &Matrix) -> Matrix {
        return self.map(&|item: f64, i: usize, j: usize| -> f64 { item * matrix.get(i, j) });
    }

    pub fn scale(&self, value: f64) -> Matrix {
        return self.map(&|_, i: usize, j: usize| -> f64 { value * self.get(i, j) });
    }

    pub fn map(&self, mapper: &dyn Fn(f64, usize, usize) -> f64) -> Matrix {
        let mut result: Vec<f64> = vec![0_f64; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[calc_index(i, j, self.cols)] = mapper(self.get(i, j), i, j)
            }
        }
        create_matrix(result, self.rows, self.cols)
    }

    pub fn iterate(&self, iterator: &mut dyn FnMut(f64, usize)) {
        for i in 0..self.rows {
            iterator(self.get(i, 0), i)
        }
    }

    pub fn unshift(&self, value: f64) -> Matrix {
        if self.cols != 1 {
            panic!("Can only unshift vertical vectors!")
        }
        let mut result: Vec<f64> = vec![0_f64; self.rows * (self.cols + 1)];
        result[0] = value;
        for i in 0..self.rows * self.cols {
            result[i + 1] = self.items[i]
        }
        create_matrix(result, self.rows + 1, self.cols)
    }

    pub fn to_arrays(&self) -> Vec<Vec<f64>> {
        let mut result: Vec<Vec<f64>> = vec![vec![0_f64; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.get(i, j)
            }
        }
        result
    }

    pub fn print(&self) {
        println!("{:?}", self);
    }
}

impl From<&Vec<f64>> for Matrix {
    fn from(items: &Vec<f64>) -> Matrix {
        let rows = items.len();
        let cols = 1;

        let mut result: Vec<f64> = vec![0_f64; rows];
        for i in 0..rows {
            result[i] = items[i]
        }
        create_matrix(result, rows, cols)
    }
}

impl From<&Vec<Vec<f64>>> for Matrix {
    fn from(items: &Vec<Vec<f64>>) -> Matrix {
        let rows = items.len();
        let cols = items[0].len();

        let mut result: Vec<f64> = vec![0_f64; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] = items[i][j]
            }
        }
        create_matrix(result, rows, cols)
    }
}

fn create_matrix(items: Vec<f64>, rows: usize, cols: usize) -> Matrix {
    Matrix { items, rows, cols }
}

fn calc_index(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}
