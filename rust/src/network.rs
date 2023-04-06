use crate::layer::Layer;
use crate::matrix::Matrix;
use crate::optimizer::Optimizer;
use crate::{activation, layer};
use crate::{
    error,
    utils::{random_weights, DataSetItem},
};

pub struct Network {
    error: error::ErrorFunctionKey,
    pub dataset: Vec<DataSetItem>,
    pub layers: Vec<Layer>,
}

pub struct ForwardResult {
    pub results: Vec<layer::ForwardPassResult>,
    pub activated_input: Matrix,
}

pub struct GradientResult {
    pub gradients: Vec<Matrix>,
    pub loss: f64,
}

pub fn create_network(error: error::ErrorFunctionKey) -> Network {
    Network {
        error,
        dataset: Vec::new(),
        layers: Vec::new(),
    }
}

impl Network {
    pub fn set_data(&mut self, dataset: Vec<DataSetItem>) {
        self.dataset = dataset;
    }

    pub fn add(&mut self, activation: activation::ActivationFunctionKey, count: u32) {
        self.layers
            .push(layer::create_layer(count, activation, self.error))
    }

    pub fn compute_gradients(&self) -> GradientResult {
        let mut total_gradient = Vec::new();
        for layer in &self.layers {
            total_gradient.push(layer.weights.scale(0.0))
        }

        let mut total_loss = 0.0;

        for data in &self.dataset {
            let data = self.compute_result(
                &Matrix::from(&data.input),
                &Matrix::from(&data.output),
                &mut total_gradient,
            );
            let results = data.results;
            total_loss += results[results.len() - 1].error / self.dataset.len() as f64
        }

        GradientResult {
            gradients: total_gradient,
            loss: total_loss,
        }
    }

    pub fn compute_result(
        &self,
        real_input: &Matrix,
        expected: &Matrix,
        total: &mut Vec<Matrix>,
    ) -> ForwardResult {
        let data = self.forward_pass(real_input, expected);

        self.backward_pass(&data.activated_input, expected, &data.results, total);

        data
    }

    pub fn forward_pass(&self, input: &Matrix, expected: &Matrix) -> ForwardResult {
        let activated_input = (activation::IDENTITY_ACTIVATION.forward)(&input, false);

        let mut results: Vec<layer::ForwardPassResult> = Vec::new();
        for layer in &self.layers {
            let current_activation = if results.len() > 0 {
                &results[results.len() - 1].activated
            } else {
                &activated_input
            };
            let result = layer.forward_pass(current_activation, &expected);
            results.push(result);
        }

        ForwardResult {
            results,
            activated_input,
        }
    }

    pub fn backward_pass(
        &self,
        input: &Matrix,
        expected: &Matrix,
        results: &Vec<layer::ForwardPassResult>,
        total: &mut Vec<Matrix>,
    ) {
        let result_out = &results[results.len() - 1];
        let layer_out = &self.layers[self.layers.len() - 1];

        let delta_out = layer_out.output_pass(result_out, &expected);
        let mut deltas = vec![delta_out];

        for i in (0..self.layers.len() - 1).rev() {
            let result = &results[i];
            let delta = self.layers[i].backward_pass(&result, &deltas[0], &self.layers[i + 1]);
            deltas.insert(0, delta);
        }

        for l in 0..total.len() {
            let weights = &mut total[l];
            let activations = if l == 0 {
                &input
            } else {
                &results[l - 1].activated
            };
            let delta = &deltas[l];

            for i in 0..weights.rows {
                for j in 0..weights.cols {
                    let diff = delta.get(i, 0) * activations.get(j, 0);
                    let new_val = weights.get(i, j) + (diff / self.dataset.len() as f64);
                    weights.set(i, j, new_val);
                }
            }
        }
    }

    pub fn predict(&self, item: &DataSetItem) -> ForwardResult {
        let input = Matrix::from(&item.input);
        let output = Matrix::from(&item.output);
        self.forward_pass(&input, &output)
    }

    pub fn initialize(&mut self, dataset: Vec<DataSetItem>, weights: Option<Vec<Vec<Vec<f64>>>>) {
        self.dataset = dataset;
        if self.layers.len() == 0 {
            panic!("No layers provided!")
        }
        let mut new_weights: Vec<Vec<Vec<f64>>> = Vec::new();

        let mut current_dim = self.dataset[0].input.len();
        if weights.is_none() {
            for layer in &self.layers {
                let rows = layer.neuron_count as usize;
                let cols = current_dim + 1;

                current_dim = rows;
                new_weights.push(random_weights(rows, cols))
            }
        } else {
            for w in weights.unwrap() {
                new_weights.push(w);
            }
        }

        let count = self.layers.len();
        for l in 0..count {
            let w = &new_weights[l];
            self.layers[l].initialize(Matrix::from(w), l == count - 1);
        }
    }
}
