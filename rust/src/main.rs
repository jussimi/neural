mod activation;
mod error;
mod layer;
mod matrix;
mod network;
mod optimizer;
mod utils;

use activation::ActivationFunctionKey;
use error::ErrorFunctionKey;
use network::create_network;
use utils::{read_lines, DataSetItem};

use crate::{
    optimizer::{sgd_optimizer, Optimizer},
    utils::{generate_batch, is_correct_category, one_hot_encode},
};

fn main() {
    // run_xor();
    run_mnist()
}

fn run_mnist() {
    const LABEL_SIZE: usize = 10;
    const BATCH_SIZE: u32 = 200;
    const EPOCHS: u64 = 40;

    fn read_data(path: &str) -> Vec<DataSetItem> {
        let mut result = Vec::new();
        if let Ok(lines) = read_lines(path.to_string()) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(inp) = line {
                    if inp.len() == 0 {
                        continue;
                    }
                    let mut items: Vec<f64> = inp
                        .split(",")
                        .map(|x| -> f64 { x.parse::<f64>().unwrap() })
                        .collect();
                    let output = one_hot_encode(items.remove(0), LABEL_SIZE);
                    result.push(DataSetItem {
                        input: items,
                        output,
                    })
                }
            }
        }
        result
    }

    fn lr_schedule(i: u64) -> f64 {
        let initial = 0.5;
        let decay = 0.1;
        let min = 0.05;

        let result = initial * (1.0 / (1.0 + decay * i as f64));

        result.max(min)
    }

    fn validate(network: &network::Network, data: &Vec<DataSetItem>) {
        let mut correct_count = 0;
        let mut total_loss = 0.0;
        for item in data {
            let result = network.predict(&item);
            let last = &result.results[result.results.len() - 1];

            let is_correct = is_correct_category(&last.activated, &item.output);
            total_loss += last.error;
            if is_correct {
                correct_count += 1;
            }
        }
        println!("\nValidation loss: {}", total_loss / data.len() as f64);
        println!(
            "     percentage: {}",
            correct_count as f64 / data.len() as f64
        );
    }

    let train_data = read_data("./mnist_train.csv");
    let test_data = read_data("./mnist_test.csv");

    let set_size = train_data.len();
    let epoch_iterations = (set_size as f64 / BATCH_SIZE as f64).floor() as u64;

    let iterations = epoch_iterations * EPOCHS;

    let mut network = create_network(ErrorFunctionKey::CrossEntropy);
    network.add(ActivationFunctionKey::Sigmoid, 32);
    network.add(ActivationFunctionKey::Softmax, 10);
    network.initialize(generate_batch(&train_data, BATCH_SIZE), None);

    use std::time::Instant;
    let now = Instant::now();

    let mut optimizer = sgd_optimizer(None, Some(lr_schedule));
    for i in 0..iterations {
        let data = network.compute_gradients();

        optimizer.do_update(&data, &mut network, i);
        network.set_data(generate_batch(&train_data, BATCH_SIZE));

        if i % 10 == 0 {
            println!("Iter {}: time taken {:.2?}", i, now.elapsed().as_millis());
            println!("   - loss: {}", data.loss);
        }

        if i > 0 && i % epoch_iterations == 0 {
            validate(&network, &test_data)
        }
    }
    validate(&network, &test_data);

    println!("Elapsed: {:.2?}", now.elapsed().as_millis());
}

fn run_xor() {
    let w1 = vec![vec![0.5, 0.5, 0.5], vec![-0.5, -0.5, -0.5]];
    let w2 = vec![vec![-0.5, 0.5, 0.5]];

    let dataset = vec![
        DataSetItem {
            input: vec![1.0, 1.0],
            output: vec![0.0],
        },
        DataSetItem {
            input: vec![1.0, 0.0],
            output: vec![1.0],
        },
        DataSetItem {
            input: vec![0.0, 1.0],
            output: vec![1.0],
        },
        DataSetItem {
            input: vec![0.0, 0.0],
            output: vec![0.0],
        },
    ];

    let mut network = create_network(ErrorFunctionKey::LogLoss);
    network.add(ActivationFunctionKey::TanH, 2);
    network.add(ActivationFunctionKey::Sigmoid, 1);
    network.initialize(dataset, Some(vec![w1, w2]));

    let mut optimizer = sgd_optimizer(Some(1.0), None);

    use std::time::Instant;
    let now = Instant::now();

    for i in 0..300 {
        let data = network.compute_gradients();
        optimizer.do_update(&data, &mut network, i);
    }

    let result = network.compute_gradients();
    println!("Elapsed: {:.2?}", now.elapsed().as_micros());
    println!("{}", result.loss);

    for l in network.layers {
        l.weights.print()
    }
}
