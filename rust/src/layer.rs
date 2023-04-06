use crate::activation;
use crate::error;
use crate::matrix::Matrix;

pub struct ForwardPassResult {
    pub sum: Matrix,
    pub activated: Matrix,
    pub error: f64,
}

pub struct Layer {
    pub is_output: bool,
    pub neuron_count: u32,
    pub weights: Matrix,
    pub weights_transpose: Matrix,
    activation_key: activation::ActivationFunctionKey,
    activation: activation::ActivationFN,
    error_key: error::ErrorFunctionKey,
    error: error::ErrorFN,
}

impl Layer {
    pub fn forward_pass(&self, input: &Matrix, expected: &Matrix) -> ForwardPassResult {
        let sum = self.weights.multiply(input);

        let activated = (self.activation.forward)(&sum, self.is_output);

        let error = if self.is_output {
            (self.error.loss)(&activated, expected)
        } else {
            -1.0
        };
        ForwardPassResult {
            sum,
            activated,
            error,
        }
    }

    pub fn backward_pass(
        &self,
        result: &ForwardPassResult,
        delta_next: &Matrix,
        layer_next: &Layer,
    ) -> Matrix {
        let weights = &layer_next.weights_transpose;
        let backward = self.activation.backward;
        weights
            .multiply(delta_next)
            .hadamard(&backward(&result.sum))
    }

    pub fn output_pass(&self, result: &ForwardPassResult, expected: &Matrix) -> Matrix {
        if let activation::ActivationFunctionKey::Softmax = self.activation_key {
            if let error::ErrorFunctionKey::CrossEntropy = self.error_key {
                return (self.activation.output)(&result.activated, expected);
            } else {
                panic!("can only use softmax with cross-entropy");
            }
        }
        let error = (self.error.grad)(&result.activated, expected);
        let backward = (self.activation.backward)(&result.sum);

        error.hadamard(&backward)
    }

    pub fn set_weights(&mut self, weights: Matrix) {
        self.weights = weights;
        self.weights_transpose = self.weights.omit(0).transpose();
    }

    pub fn initialize(&mut self, weights: Matrix, is_output: bool) {
        self.set_weights(weights);
        self.is_output = is_output;
    }
}

pub fn create_layer(
    count: u32,
    activation: activation::ActivationFunctionKey,
    error: error::ErrorFunctionKey,
) -> Layer {
    let error_fn = match error {
        error::ErrorFunctionKey::CrossEntropy => error::CE_LOSS,
        error::ErrorFunctionKey::LogLoss => error::LOG_LOSS,
        error::ErrorFunctionKey::MeanSquared => error::MSE_LOSS,
    };
    let activation_fn = match activation {
        activation::ActivationFunctionKey::ReLu => activation::RELU_ACTIVATION,
        activation::ActivationFunctionKey::Sigmoid => activation::SIGMOID_ACTIVATION,
        activation::ActivationFunctionKey::TanH => activation::TANH_ACTIVATION,
        activation::ActivationFunctionKey::Softmax => activation::SOFTMAX_ACTIVATION,
    };
    Layer {
        is_output: false,
        activation: activation_fn,
        activation_key: activation,
        error: error_fn,
        error_key: error,
        neuron_count: count,
        weights: Matrix::from(&vec![0.0]),
        weights_transpose: Matrix::from(&vec![0.0]),
    }
}
