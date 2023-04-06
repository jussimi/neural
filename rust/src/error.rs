use crate::matrix::Matrix;

#[derive(Copy, Clone)]
pub enum ErrorFunctionKey {
    CrossEntropy,
    MeanSquared,
    LogLoss,
}

pub struct ErrorFN {
    pub loss: fn(output: &Matrix, expected: &Matrix) -> f64,
    pub grad: fn(output: &Matrix, expected: &Matrix) -> Matrix,
}

pub const LOG_LOSS: ErrorFN = ErrorFN {
    loss: |output, expected| -> f64 {
        let t = output.get(0, 0);
        let y = expected.get(0, 0);
        -(y * t.ln() + (1.0 - y) * (1.0 - t).ln())
    },
    grad: |output, expected| -> Matrix {
        let t = output.get(0, 0);
        let y = expected.get(0, 0);
        let result = (t - y) / (t - t.powi(2));
        Matrix::from(&vec![result])
    },
};

pub const CE_LOSS: ErrorFN = ErrorFN {
    loss: |output, expected| -> f64 {
        let result = output
            .map(&|x, _, _| -> f64 { x.ln() })
            .transpose()
            .multiply(expected);
        -result.get(0, 0)
    },
    grad: |output, expected| -> Matrix {
        output.map(&|t, i, _| -> f64 {
            let y = expected.get(i, 0);
            -(y / t)
        })
    },
};

pub const MSE_LOSS: ErrorFN = ErrorFN {
    loss: |output, expected| -> f64 {
        let mut sum = 0.0;

        for i in 0..output.rows {
            let diff = output.get(i, 0) - expected.get(i, 0);
            sum += diff.powi(2);
        }
        sum
    },
    grad: |output, expected| -> Matrix { output.map(&|x, i, j| -> f64 { x - expected.get(i, j) }) },
};
