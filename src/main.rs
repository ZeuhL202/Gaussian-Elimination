mod matrix;

use matrix::Matrix;
use matrix::fraction::Fraction;

fn main() {
    let a = vec![
        10,  1, -1,  1,  55,
         2,  9, -2,  4, -76,
        -1, -3, 12,  1,  75,
        -1,  2, -1, 15, -58
    ]
    .into_iter()
    .map(|n| Fraction::new(n, 1))
    .collect::<Vec<Fraction>>();

    let mut matrix = Matrix::new(&a);
    println!("{}", matrix);
    let ans = matrix.jacobi_iterative(35, true);
    println!("ans: {:?}", ans);
}