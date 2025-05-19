use zeuhl_sem::SEM;

fn main() {
    let mut sem = SEM::new_from_deno1(
        vec![
            4, 1, 1, 12,
            1, 4, 1, 15,
            1, 1, 4,  9,
        ]
    );

    sem.gauss_seidel(0.01, true);
}