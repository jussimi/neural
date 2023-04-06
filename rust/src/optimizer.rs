use crate::{matrix::Matrix, network};

pub trait Optimizer {
    fn do_update(
        &mut self,
        gradient_result: &network::GradientResult,
        network: &mut network::Network,
        iteration: u64,
    );
}

pub struct GradientDescentOptimizer {
    pub learning_rate: Option<f64>,
    pub learning_rate_schedule: Option<fn(u64) -> f64>,

    pub momentum: f64,

    is_nesterov: bool,
    current_weights: Vec<Matrix>,

    velocity: Vec<Matrix>,

    initialized: bool,
}

impl GradientDescentOptimizer {
    fn initialize(&mut self, result: &network::GradientResult) {
        self.velocity = Vec::new();
        for grad in &result.gradients {
            self.velocity.push(grad.scale(0.0))
        }
        self.initialized = true;
    }

    fn update_velocity(&mut self, result: &network::GradientResult, learning_rate: f64) {
        let gradients = &result.gradients;

        let mut new_vel = Vec::new();
        for i in 0..self.velocity.len() {
            let layer_vel = self.velocity[i]
                .scale(self.momentum)
                .subtract(&gradients[i].scale(learning_rate));
            new_vel.push(layer_vel);
        }
        self.velocity = new_vel;
    }
}

impl Optimizer for GradientDescentOptimizer {
    fn do_update(
        &mut self,
        result: &network::GradientResult,
        network: &mut network::Network,
        iteration: u64,
    ) {
        if !self.initialized {
            self.initialize(result)
        }
        let lr = get_learning_rate(self.learning_rate, self.learning_rate_schedule, iteration);

        self.update_velocity(result, lr);

        for i in 0..network.layers.len() {
            let layer = &mut network.layers[i];
            let weights = &layer.weights;

            layer.set_weights(weights.sum(&self.velocity[i]))
        }
    }
}

pub fn sgd_optimizer(
    learning_rate: Option<f64>,
    schedule: Option<fn(u64) -> f64>,
) -> GradientDescentOptimizer {
    GradientDescentOptimizer {
        learning_rate,
        learning_rate_schedule: schedule,
        momentum: 0.0,
        is_nesterov: false,
        current_weights: Vec::new(),
        velocity: Vec::new(),
        initialized: false,
    }
}

fn get_learning_rate(lr: Option<f64>, schedule: Option<fn(u64) -> f64>, iteration: u64) -> f64 {
    if schedule.is_some() {
        return (schedule.unwrap())(iteration);
    }
    if lr.is_some() {
        return lr.unwrap();
    }
    panic!("Invalid learning rate supplied")
}
