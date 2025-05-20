use zeuhl_sem::SEM;

fn main() {
    let mut sem = SEM::new_from_deno1(
        vec![
            -5,  -2,  3,  1,  12,
             2, -10, -3, -6,  35,
            -1,  -8, -9, -5, -10,
             1,   4, -1,  7, -26
        ]
    );

    sem.sor(1.9, 0.00001, true);
}