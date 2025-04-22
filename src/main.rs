mod gauss;
use gauss::Gauss;
fn main() {
    let a = vec![
        1,  1, 5,
        2, -1, 1,
    ];

    let mut gauss = Gauss::new_numerator(a);
    println!("{}", gauss);
    gauss.solve(true);
    println!("{}", gauss);
}