use crate::matrix::Matrix;

struct ScalarActivationFN {
    forward: fn(f64) -> f64,
    backward: fn(f64) -> f64,
}

pub enum ActivationFunctionKey {
    Sigmoid,
    TanH,
    ReLu,
    Softmax,
}

const SCALAR_RELU: ScalarActivationFN = ScalarActivationFN {
    forward: |x| -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    },
    backward: |x| -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    },
};

fn tanh_fn(x: f64) -> f64 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

const SCALAR_TANH: ScalarActivationFN = ScalarActivationFN {
    forward: tanh_fn,
    backward: |x| -> f64 { 1.0 - tanh_fn(x).powi(2) },
};

fn sigmoid_fn(x: f64) -> f64 {
    let divisor = 1.0 + (-x).exp();
    1.0 / divisor
}
const SCALAR_SIGMOID: ScalarActivationFN = ScalarActivationFN {
    forward: sigmoid_fn,
    backward: |x| -> f64 { sigmoid_fn(x) * (1.0 - sigmoid_fn(x)) },
};

pub struct ActivationFN {
    pub forward: fn(sum: &Matrix, isOutput: bool) -> Matrix,
    pub backward: fn(sum: &Matrix) -> Matrix,
    pub output: fn(output: &Matrix, expected: &Matrix) -> Matrix,
}

pub const RELU_ACTIVATION: ActivationFN = ActivationFN {
    forward: |sum, is_output| -> Matrix { activate(SCALAR_RELU.forward, sum, !is_output) },
    backward: |sum| -> Matrix { backpropagate(SCALAR_RELU.backward, sum) },
    output: |sum, _| -> Matrix { backpropagate(SCALAR_RELU.forward, sum) },
};

pub const TANH_ACTIVATION: ActivationFN = ActivationFN {
    forward: |sum, is_output| -> Matrix { activate(SCALAR_TANH.forward, sum, !is_output) },
    backward: |sum| -> Matrix { backpropagate(SCALAR_TANH.backward, sum) },
    output: |sum, _| -> Matrix { backpropagate(SCALAR_TANH.forward, sum) },
};

pub const SIGMOID_ACTIVATION: ActivationFN = ActivationFN {
    forward: |sum, is_output| -> Matrix { activate(SCALAR_SIGMOID.forward, sum, !is_output) },
    backward: |sum| -> Matrix { backpropagate(SCALAR_SIGMOID.backward, sum) },
    output: |sum, _| -> Matrix { backpropagate(SCALAR_SIGMOID.forward, sum) },
};

pub const IDENTITY_ACTIVATION: ActivationFN = ActivationFN {
    forward: |sum, is_output| -> Matrix { activate(|x| -> f64 { x }, sum, !is_output) },
    backward: |_| -> Matrix { panic!("use identity only on first layer") },
    output: |_, _| -> Matrix { panic!("use identity only on first layer") },
};

pub const SOFTMAX_ACTIVATION: ActivationFN = ActivationFN {
    forward: |sum, is_output| -> Matrix {
        let result = softmax_fn(sum);
        if !is_output {
            return result.unshift(1.0);
        }
        result
    },
    output: |sum, expected| -> Matrix { sum.subtract(expected) },
    backward: |_| -> Matrix { panic!("use softmax only on output") },
};

fn activate(mapper: fn(f64) -> f64, sum: &Matrix, append_bias: bool) -> Matrix {
    let vect = sum.map(&|x, _, _| -> f64 { mapper(x) });
    if append_bias {
        return vect.unshift(1.0);
    }
    vect
}

fn backpropagate(mapper: fn(f64) -> f64, sum: &Matrix) -> Matrix {
    sum.map(&|x, _, _| -> f64 { mapper(x) })
}

fn softmax_fn(vector: &Matrix) -> Matrix {
    let mut max = vector.get(0, 0);
    vector.iterate(&mut |value, _| {
        if value > max {
            max = value
        }
    });

    let mut sum = 0.0;
    vector.iterate(&mut |value, _| {
        sum += (value - max).exp();
    });

    vector.map(&|x, _, _| -> f64 { (x - max).exp() / sum })
}
