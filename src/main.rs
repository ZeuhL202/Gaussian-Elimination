mod gauss;

use gauss::Gauss;
use gauss::fraction::Fraction;

fn main() {
    let a = vec![
        (1, 1), ( 1, 1), (5, 1),
        (2, 1), (-1, 1), (1, 1),
    ]
    .into_iter()
    .map(|(n, d)| Fraction::new(n, d))
    .collect::<Vec<Fraction>>();

    let mut gauss = Gauss::new(a);
    println!("{}", gauss);
    gauss.solve(true);
    println!("{}", gauss);
}