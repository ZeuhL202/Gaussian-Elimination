use std::{fmt, vec};
use zeuhl_fraction::Fraction;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SEM {
    pub unknown_num: usize,
    pub value: Vec<Vec<Fraction>>,
}

impl SEM {
    /// calculates the unknown_num of the sem from the number of term
    /// 連立方程式の未知数を求める
    ///
    /// i.pow(n): number of term on the left  side
    /// i:        number of term on the right side
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    /// let sem = SEM::new_from_deno1(vec![
    ///     1,  1,  1,  6,
    ///     2, -1,  3,  7,
    ///     1,  4, -1, 10
    /// ]);
    ///
    /// // term_len = 12
    ///
    /// // i = 0  (i.pow(2) + i) =  0  (Greater than 0)
    /// // i = 1  (i.pow(2) + i) =  2  (Greater than 0)
    /// // i = 2  (i.pow(2) + i) =  6  (Greater than 0)
    /// // i = 3  (i.pow(2) + i) = 12  (Equal   to   0)
    ///
    /// // unknown_num = 3
    /// assert_eq!(sem.unknown_num, 3);
    /// ```
    fn calculate_unknown_num(fractions: &[Fraction]) -> usize {
        let frac_len = fractions.len();
        let mut i: usize = 0;

        loop {
            let frac_left = frac_len - i.pow(2) - i;

            if frac_left == 0 {
                break i;
            } else if frac_left > 0 {
                i += 1;
            } else {
                // painc because the input is not solvable sem
                panic!("Invalid input: not a valid unknown_num");
            }
        }
    }

    /// Creates a new SEM object from a vector of fractions
    /// 連立方程式を分数から作成する
    ///
    /// # Examples
    /// ```
    /// use zeuhl_fraction::Fraction;
    /// use zeuhl_sem::SEM;
    ///
    /// let fractions: Vec<Fraction> =
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1
    ///     ]
    ///     .iter_mut()
    ///     .map(|x| Fraction::new(*x, 1))
    ///     .collect();
    ///
    /// let sem = SEM::new_from_frac(&fractions);
    /// ```
    pub fn new_from_frac(fractions: &[Fraction]) -> Self {
        let unknown_num = Self::calculate_unknown_num(fractions);

        let value = fractions
            .chunks(unknown_num + 1)
            .map(|x| x.to_vec())
            .collect();

        Self { unknown_num, value }
    }

    /// Creates a new SEM object from a vector of (isize, usize)
    /// 連立方程式を (isize, usize) のベクターから作成する
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let fractions: Vec<(isize, usize)> =
    ///     vec![
    ///         (1, 1), ( 1, 2), (5, 9),
    ///         (2, 2), (-1, 4), (1, 5)
    ///     ];
    ///
    /// let sem = SEM::new_from_tuple(fractions);
    /// ```
    pub fn new_from_tuple(vec: Vec<(isize, usize)>) -> Self {
        let fractions = vec
            .clone()
            .iter_mut()
            .map(|(n, d)| Fraction::new(*n, *d))
            .collect::<Vec<Fraction>>();

        Self::new_from_frac(&fractions)
    }

    /// Creates a new SEM object from a vector of isize
    /// 連立方程式を分母1の分数から作成する。
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let fractions: Vec<isize> =
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1
    ///     ];
    ///
    /// let sem = SEM::new_from_deno1(fractions);
    /// ```
    pub fn new_from_deno1(vec: Vec<isize>) -> Self {
        let fractins = vec
            .clone()
            .iter_mut()
            .map(|x| Fraction::new(*x, 1))
            .collect::<Vec<Fraction>>();

        Self::new_from_frac(&fractins)
    }

    /// Solves the sem using gaussian elimination
    /// 連立方程式をガウスの消去法で解く
    ///
    /// # Examples
    /// ```
    /// use zeuhl_fraction::Fraction;
    /// use zeuhl_sem::SEM;
    ///
    /// let mut SEM = SEM::new_from_deno1(
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1,
    ///     ]
    /// );
    /// SEM.gaussian_elimination(false);
    ///
    /// let answer: Vec<Fraction> =
    ///     vec![2, 3]
    ///     .iter_mut()
    ///     .map(|x| Fraction::new(*x, 1))
    ///     .collect();
    ///
    /// assert_eq!(SEM.extract(), answer);
    /// ```
    pub fn gaussian_elimination(&mut self, debug: bool) {
        // Use forward elimination to zero out the lower triangle
        // 前進消去で下三角行列を0にする
        for i in 0..self.unknown_num {
            // Set the diagonal elements to 1
            // 対角要素を1にする
            let pivot = self.value[i][i];
            self.value[i]
                .iter_mut()
                .for_each(|x| *x /= pivot);

            // Do forward elimination
            // 前進消去
            self.forward_elimination(i, debug);
        }

        // Use backward substitution to zero out the upper triangle
        // 後退代入で上三角行列を0にする
        for i in 1..self.unknown_num {
            self.backward_substitution(i, debug);
        }
    }

    /// Performs direct transformation on the sem
    /// 直接変換を行う
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1,
    ///     ]
    /// );
    ///
    /// sem.direct_transformation(0, 1);
    ///
    /// let ans_sem = SEM::new_from_deno1(
    ///     vec![
    ///         1,  1,  5,
    ///         0,  -3, -9,
    ///     ]
    /// );
    ///
    /// assert_eq!(sem, ans_sem);
    /// ```
    pub fn direct_transformation(&mut self, pivot_index: usize, target_row: usize) {
        let factor = self.value[target_row][pivot_index];

        for i in pivot_index..=self.unknown_num {
            let value_pivot_index = self.value[pivot_index][i];
            self.value[target_row][i] -= value_pivot_index * factor;
        }
    }

    /// Performs forward elimination on the sem
    /// 前進消去を1列分行う
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         5, 10, 15, 5,
    ///         2, -1,  1, 2,
    ///         1,  1,  1, 3
    ///     ]
    /// );
    ///
    /// let pivot = sem.value[0][0];
    /// sem.value[0]
    ///     .iter_mut()
    ///     .for_each(|x| *x /= pivot);
    ///
    /// sem.forward_elimination(0, false);
    ///
    /// let ans_sem = SEM::new_from_deno1(
    ///     vec![
    ///         1,  2,  3, 1,
    ///         0, -5, -5, 0,
    ///         0, -1, -2, 2,
    ///     ]
    /// );
    ///
    /// assert_eq!(sem, ans_sem);
    /// ```
    pub fn forward_elimination(&mut self, pivot_index: usize, debug: bool) {
        for i in pivot_index+1..self.unknown_num {
            self.direct_transformation(pivot_index, i);
        }

        if debug {
            print!("forward_elimination: pivot_num = {}\n{}\n", pivot_index, self);
        }
    }

    /// Performs backward substitution on the sem
    /// 後退代入を1列分行う
    ///
    /// # Examples
    ///
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         1, 0, 4,  6,
    ///         0, 1, 2, -7,
    ///         0, 0, 1,  1,
    ///     ]
    /// );
    ///
    /// sem.backward_substitution(2, false);
    ///
    /// let ans_sem = SEM::new_from_deno1(
    ///     vec![
    ///         1, 0, 0,  2,
    ///         0, 1, 0, -9,
    ///         0, 0, 1,  1,
    ///     ]
    /// );
    ///
    /// assert_eq!(sem, ans_sem);
    /// ```
    pub fn backward_substitution(&mut self, pivot_index: usize, debug: bool) {
        for i in 0..pivot_index {
            self.direct_transformation(pivot_index, i);
        }

        if debug {
            print!("backward_elimination: pivot_num = {}\n{}\n", pivot_index, self);
        }
    }

    fn iterative_model(
        &mut self,
        func: fn(&mut SEM, f64, &mut Vec<f64>, &mut Vec<f64>),
        omega: f64,
        convergence_conditions: f64,
        debug: bool
    ) -> Vec<isize> {

        let mut x_before  = vec![0f64; self.unknown_num];
        let mut x_current = vec![0f64; self.unknown_num];

        if debug {
            println!("i   0: {}", "         0 ".repeat(self.unknown_num));
        }

        let mut attempts = 0;
        loop {
            if attempts > 10000 {
                println!("Failed to converge");
                break;
            }

            func(self, omega, &mut x_current, &mut x_before);

            if {
                let mut sum = 0f64;
                for i in 0..self.unknown_num {
                    sum += (x_current[i] - x_before[i]).powi(2);
                }
                sum.sqrt() < convergence_conditions
            } {
                break
            }

            if debug {
                print!("i {:3}:", attempts+1);
                for j in 0..self.unknown_num {
                    print!(" {:>10}", (x_current[j] * 100000000.0).round() / 100000000.0);
                }
                println!();
            }

            x_before = x_current.clone();
            attempts += 1;
        }

        x_current
            .iter()
            .map(|x| x.round() as isize)
            .collect()
    }

    /// Solves the sem using jacobi iterative method
    /// ヤコビ法を用いて解く
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         4, 1, 1, 12,
    ///         1, 4, 1, 15,
    ///         1, 1, 4,  9,
    ///     ],
    /// );
    ///
    /// let result = sem.jacobi_iterative(0.01, false);
    ///
    /// assert_eq!(result, vec![2, 3, 1]);
    /// ```
    pub fn jacobi_iterative(&mut self, convergence_conditions: f64, debug: bool) -> Vec<isize> {
        fn func (
            sem: &mut SEM,
            _: f64,
            x_current: &mut Vec<f64>,
            x_before: &mut Vec<f64>
        ) {
            for i in 0..sem.unknown_num {
                let mut sum = 0f64;

                for j in 0..sem.unknown_num {
                    if i != j {
                        sum += sem.value[i][j].as_f64() * x_before[j];
                    }
                }

                x_current[i] = (sem.value[i][sem.unknown_num].as_f64() - sum) / sem.value[i][i].as_f64();
            }
        }

        self.iterative_model(func, 0.0, convergence_conditions, debug)
    }

    /// Solves the sem using gauss seidel iterative method
    /// ガウスセイデル法を用いて解く
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         4, 1, 1, 12,
    ///         1, 4, 1, 15,
    ///         1, 1, 4,  9,
    ///     ],
    /// );
    ///
    /// let result = sem.gauss_seidel(0.01, false);
    ///
    /// assert_eq!(result, vec![2, 3, 1]);
    /// ```
    pub fn gauss_seidel(&mut self, convergence_conditions: f64, debug: bool) -> Vec<isize> {
        fn func (
            sem: &mut SEM,
            _: f64,
            x_current: &mut Vec<f64>,
            x_before: &mut Vec<f64>
        ) {
            for i in 0..sem.unknown_num {
                let mut ans = 0f64;

                for j in 0..=sem.unknown_num {
                    if j == i {
                        continue
                    }

                    ans += sem.value[i][j].as_f64() * {
                        if j == sem.unknown_num {
                            1f64
                        } else if j < i {
                            -x_current[j]
                        } else {
                            -x_before[j]
                        }
                    };
                }

                ans /= sem.value[i][i].as_f64();
                x_current[i] = ans;
            }
        }

        self.iterative_model(func, 0.0, convergence_conditions, debug)
    }

    /// Solves the sem using sor iterative method
    /// SOR法を用いて解く
    ///
    /// # Examples
    /// ```
    /// use zeuhl_sem::SEM;
    ///
    /// let mut sem = SEM::new_from_deno1(
    ///     vec![
    ///         4, 1, 1, 12,
    ///         1, 4, 1, 15,
    ///         1, 1, 4,  9,
    ///     ],
    /// );
    ///
    /// let result = sem.sor(1.25, 0.01, false);
    ///
    /// assert_eq!(result, vec![2, 3, 1]);
    /// ```
    pub fn sor(&mut self, omega: f64, convergence_conditions: f64, debug: bool) -> Vec<isize> {
        fn func (
            sem: &mut SEM,
            omega: f64,
            x_current: &mut Vec<f64>,
            x_before: &mut Vec<f64>
        ){
            for i in 0..sem.unknown_num {
                let mut ans = 0f64;

                for j in 0..=sem.unknown_num {
                    if j == i {
                        continue
                    }

                    ans += sem.value[i][j].as_f64() * {
                        if j == sem.unknown_num {
                            1f64
                        } else if j < i {
                            -x_current[j]
                        } else {
                            -x_before[j]
                        }
                    };
                }

                ans /= sem.value[i][i].as_f64();
                ans *= omega;

                ans += (1.0 - omega) * x_current[i];

                x_current[i] = ans;
            }
        }

        self.iterative_model(func, omega, convergence_conditions, debug)
    }

    pub fn extract(&self) -> Vec<Fraction> {
        let mut result = vec![];

        for i in 0..self.unknown_num {
            result.push(self.value[i][self.unknown_num]);
        }

        result
    }
}

fn max_space_each_col(sem: Vec<Vec<Fraction>>) -> Vec<usize> {
    let col_len = sem[0].len();
    let row_len = sem.len();

    let mut space_each_col = [0].repeat(col_len);

    for col in 0..col_len {
        for row in 0..row_len {
            let len = sem[row][col].len();

            if len > space_each_col[col] {
                space_each_col[col] = len;
            }
        }
    }

    space_each_col
}

impl fmt::Display for SEM {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_space_each_col = max_space_each_col(self.value.clone());

        // Space between parentheses on lines that don't display fractions
        //
        // # Examples
        //
        // ┌#.#..##.#┐ <- here
        // │ 1  1: 5 │
        // │ 2 -1: 1 │
        // └#.#..##.#┘ <- and here
        //
        // ┌#..#.#..##.#┐
        // │  1 2  8: 1 │
        // │ -1 5  3: 3 │
        // │  1 1 10: 0 │
        // └#..#.#..##.#┘
        //
        // (.: Space to put where a number is)
        // max_space_each_col
        //     .iter()
        //     .sum::<usize>()
        //
        // (#: Space between numbers)
        // self.unknown_num + 3
        //
        let edge_space = " ".repeat(max_space_each_col.iter().sum::<usize>() + self.unknown_num + 3);


        let each_space = |i: usize, j: usize| -> String {
            " ".repeat(max_space_each_col[j] - self.value[i][j].len() + 1)
        };

        let mut str = String::new();

        str += format!("┌{}┐\n", edge_space).as_str();

        for i in 0..self.unknown_num {
            str += "│";

            for j in 0..self.unknown_num {
                str += each_space(i, j).as_str();
                str += self.value[i][j].to_string().as_str();
            }

            str += ":";
            str += each_space(i, self.unknown_num).as_str();
            str += self.value[i][self.unknown_num].to_string().as_str();
            str += " │\n";
        }

        str += "└";
        str += edge_space.as_str();
        str += "┘";

        write!(f, "{}", str)
    }
}

// TESTS

#[cfg(test)]
mod tests {
    use super::*;

    fn deno_1_fraction(value: Vec<isize>) -> Vec::<Fraction> {
        value
            .into_iter()
            .map(|x| Fraction::new(x, 1))
            .collect()
    }

    fn gauss_temp(
        value: Vec<isize>,
        solution: Vec<isize>,
    ) {
        let mut sem = SEM::new_from_deno1(value);
        sem.gaussian_elimination(false);

        let solution = deno_1_fraction(solution);

        assert_eq!(sem.extract(), solution);
    }

    #[test]
    fn gauss_two() {
        gauss_temp(
            vec![
                1,  1, 5,
                2, -1, 1
            ],
            vec![2, 3]
        );
    }

    #[test]
    fn gauss_three() {
        gauss_temp(
            vec![
                1,  1,  1,  6,
                2, -1,  3,  7,
                1,  4, -1, 10,
            ],
            vec![3, 2, 1]
        );
    }

    #[test]
    fn gauss_four() {
        gauss_temp(
            vec![
                1,  1,  1,  1, 10,
                2,  3, -1,  1,  9,
                1, -1,  2,  3, 17,
                4,  1,  1, -2,  1
            ],
            vec![1, 2, 3, 4]
        );
    }

    fn jakobi_temp(
        value: Vec<isize>,
        solution: Vec<isize>,
    ) {
        let mut sem = SEM::new_from_deno1(value);
        let ans = sem.jacobi_iterative(0.01,false);

        assert_eq!(ans, solution);
    }

    #[test]
    fn jakobi_three() {
        jakobi_temp(
            vec![
                4, 1, 1, 12,
                1, 4, 1, 15,
                1, 1, 4,  9,
            ],
            vec![2, 3, 1],
        );
    }

    #[test]
    fn jakobi_four() {
        jakobi_temp(
            vec![
                10,  1, -1,  1,  55,
                02,  9, -2,  4, -76,
                -1, -3, 12,  1,  75,
                -1,  2, -1, 15, -58
            ],
            vec![7, -8, 5, -2],
        );
    }
}