use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct DataSetItem {
    pub input: Vec<f64>,
    pub output: Vec<f64>,
}

pub fn one_hot_encode(label: f64, len: usize) -> Vec<f64> {
    let mut vector = vec![0.0; len];
    vector[label.round() as usize] = 1.0;
    vector
}

pub fn is_correct_category(estimate: &Matrix, output: &Vec<f64>) -> bool {
    let label = output.iter().position(|x| x.round() == 1.0).unwrap();

    let index_of_max = estimate
        .items
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();

    label == index_of_max
}

pub fn generate_batch(dataset: &Vec<DataSetItem>, size: u32) -> Vec<DataSetItem> {
    let mut result: Vec<DataSetItem> = Vec::new();

    let mut used_indices: HashMap<usize, bool> = HashMap::new();

    while result.len() < size as usize {
        'inner: loop {
            let next_i: usize = rand::thread_rng().gen_range(0..dataset.len());
            if used_indices.contains_key(&next_i) {
                continue;
            }
            used_indices.insert(next_i, true);
            result.push(dataset[next_i].clone());
            break 'inner;
        }
    }
    result
}

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn random_weights(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();
    for _ in 0..rows {
        let mut row = Vec::new();
        row.push(0.0);
        for _ in 1..cols {
            row.push(rand::thread_rng().gen_range(-0.5..0.5))
        }
        result.push(row)
    }
    result
}
