mod lib;
use lib::Matrix;
use zeuhl_fraction::Fraction;

fn main() {
    let a = &[
        (4, 1), (1, 1), (1, 1), (12, 1),
        (1, 1), (4, 1), (1, 1), (15, 1),
        (1, 1), (1, 1), (4, 1), ( 9, 1)
    ]
    .iter_mut()
    .map(|(n, d)| Fraction::new(*n, *d))
    .collect::<Vec<Fraction>>();

    let mut b = Matrix::new(a);

    b.gauss_seidel(10, true);
}