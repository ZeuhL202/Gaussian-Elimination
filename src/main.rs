mod gauss;
use gauss::Gauss;

fn main() {
    let a = [
        "2/1",  "4/1",  "-4/1",  "6/1",
        "1/1", "-1/1",   "3/1",  "4/1",
        "2/1",  "3/1",  "-5/1", "11/1",
    ];

    let b = a.map(|s| s.to_string());

    let mut gauss = Gauss::new(b);
    println!("{}", gauss);
    gauss.solve();
    println!("{}", gauss);
}
