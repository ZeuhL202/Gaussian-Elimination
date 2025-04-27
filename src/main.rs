mod gauss;

use gauss::Gauss;
use gauss::fraction::Fraction;

fn main() {
    let a = vec![
        1,  1,  1,  1, 10,
        2,  3, -1,  1,  9,
        1, -1,  2,  3, 17,
        4,  1,  1, -2,  1
    ]
    .into_iter()
    .map(|n| Fraction::new(n, 1))
    .collect::<Vec<Fraction>>();

    let mut gauss = Gauss::new(&a);
    println!("{}", gauss);
    gauss.solve(true);
    gauss.extract().iter().for_each(|x| {
        print!("{} ", x);
    });
}