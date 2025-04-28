use std::fmt;
pub mod fraction;
use fraction::Fraction;

#[derive(Debug, Clone)]
pub struct Matrix {
    dimension: usize,
    value: Vec<Vec<Fraction>>,
}

impl Matrix {

    /// calculates the dimension of the matrix from the number of elements
    ///
    /// i.pow(n): number of elements on the left  side
    /// i:        number of elements on the right side
    ///
    /// # Examples
    /// ```
    /// [ 1  1  1 |  6 ]
    /// [ 2 -1  3 |  7 ]
    /// [ 1  4 -1 | 10 ]
    ///
    /// elements_len = 12
    ///
    /// i = 0  (i.pow(2) + i) =  0  (Greater than 0)
    /// i = 1  (i.pow(2) + i) =  2  (Greater than 0)
    /// i = 2  (i.pow(2) + i) =  6  (Greater than 0)
    /// i = 3  (i.pow(2) + i) = 12  (Equal   to   0)
    ///
    /// dimension = 3
    /// ```
    fn calculate_dimension(fractions: &[Fraction]) -> usize {
        let elements_len = fractions.len();
        let mut i: usize = 0;

        loop {
            let lefted_elements = elements_len - i.pow(2) - i;

            if lefted_elements == 0 {
                break i;
            } else if lefted_elements > 0 {
                i += 1;
            } else {
                // painc because the input is not solvable matrix
                panic!("Invalid input: not a valid dimension");
            }
        }
    }

    /// Creates a new Matrix object from a vector of fractions
    ///
    /// # Examples
    /// ```
    /// use Matrix::Matrix::Matrix;
    /// use Matrix::fraction::Fraction;
    ///
    /// let Matrix = Matrix::new(
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1
    ///     ]
    ///     .map(|x| Fraction::new(x, 1))
    ///     .collect()
    /// );
    /// ```
    pub fn new(fractions: &[Fraction]) -> Self {
        let dimension = Self::calculate_dimension(fractions);

        let value = fractions
            .chunks(dimension + 1)
            .map(|x| x.to_vec())
            .collect();

        Self { dimension, value }
    }

    /// Solves the matrix using gaussian elimination
    ///
    /// # Examples
    /// ```
    /// use Matrix::Matrix::Matrix;
    /// use Matrix::fraction::Fraction;
    ///
    /// let mut Matrix = Matrix::new(
    ///     vec![
    ///         1,  1, 5,
    ///         2, -1, 1
    ///     ]
    ///     .map(|x| Fraction::new(x, 1))
    ///     .collect()
    /// );
    /// Matrix.solve(false);
    /// Matrix.extract().iter().for_each(|x| {
    ///     print!("{} ", x);
    /// });
    ///
    /// assert_eq!(Matrix.extract(), vec![2, 3]);
    /// ```
    pub fn gaussian_elimination(&mut self, debug: bool) {

        // Use forward elimination to zero out the lower triangle
        for i in 0..self.dimension {
            self.forward_elimination(i);
            if debug {
                print!("forward_elimination: pivot_num = {}\n{}\n", i, self);
            }
        }

        // Use backward substitution to zero out the upper triangle
        for i in 1..self.dimension {
            for j in 0..i {
                self.backward_substitution(i, j);
            }

            if debug {
                print!("backward_elimination: pivot_num = {}\n{}\n", i, self);
            }
        }
    }

    /// Performs forward elimination on the matrix
    ///
    /// # Examples
    /// ```
    /// [ 5  10  | 15 ]
    /// [ 2  -1  |  1 ]
    ///
    /// forward_elimination(0);
    ///
    /// this is the pivot
    ///   v
    /// [ 5  10  | 15 ]
    ///
    /// divide the pivot by itself
    /// [ 1  2   |  3 ]
    ///
    /// subtract the pivot from the other rows
    /// [ 1  2   |  3 ]
    /// [ 0 -5   | -7 ]
    /// ```
    fn forward_elimination(&mut self, pivot_index: usize) {
        let pivot = self.value[pivot_index][pivot_index];

        self.value[pivot_index]
            .iter_mut()
            .for_each(|x| *x /= pivot);

        for i in pivot_index + 1..self.dimension {

            let factor = self.value[i][pivot_index];

            for j in 0..=self.dimension {
                let value_pivot_index = self.value[pivot_index][j];

                self.value[i][j] -= factor * value_pivot_index;
            }

        }
    }

    /// Performs backward substitution on the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// [ 1  3  4 |  6 ]
    /// [ 0  1  5 | -7 ]
    /// [ 0  0  1 |  1 ]
    ///
    /// backward_substitution(1, 0);
    ///
    /// [ 1 3  4 |  6 ]
    ///     ^  ^    ^     minus
    /// [ 0 1  2 | -7 ] * 3
    ///
    ///
    /// [ 1 0 -2 | 27 ]
    ///
    fn backward_substitution(&mut self, pivot_index: usize, target_row: usize) {
        for i in pivot_index..self.dimension {
            let factor = self.value[target_row][i];

            for j in 0..=self.dimension {
                let value_pivot_index = self.value[i][j];

                self.value[target_row][j] -= factor * value_pivot_index;
            }
        }
    }

    pub fn jacobi_iterative(&mut self, attempts: usize, debug: bool) -> Vec<isize> {
        let mut x_before  = vec![0f64; self.dimension];
        let mut x_current = vec![0f64; self.dimension];

        if debug {
            println!("i 0:");
        }

        for i in 0..=attempts {
            for j in 0..self.dimension {
                let mut sum = 0f64;

                for k in 0..self.dimension {
                    if j != k {
                        sum += self.value[j][k].as_f64() * x_before[k];
                    }
                }

                x_current[j] = (self.value[j][self.dimension].as_f64() - sum) / self.value[j][j].as_f64();
            }

            x_before = x_current.clone();

            if debug {
                print!("i {:3}:", i+1);
                for j in 0..self.dimension {
                    print!(" {:>20} ", x_current[j]);
                }
                println!();
            }
        }
        x_current
            .iter()
            .map(|x| x.round() as isize)
            .collect()
    }

    pub fn extract(&self) -> Vec<Fraction> {
        let mut result = vec![];

        for i in 0..self.dimension {
            result.push(self.value[i][self.dimension]);
        }

        result
    }
}

fn max_space_each_col(matrix: Vec<Vec<Fraction>>) -> Vec<usize> {
    let col_len = matrix[0].len();
    let row_len = matrix.len();

    let mut space_each_col = [0].repeat(col_len);

    for col in 0..col_len {
        for row in 0..row_len {
            let len = matrix[row][col].len();

            if len > space_each_col[col] {
                space_each_col[col] = len;
            }
        }
    }

    space_each_col
}

impl fmt::Display for Matrix {

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
        // self.dimension + 3
        //
        let edge_space = " ".repeat(max_space_each_col.iter().sum::<usize>() + self.dimension + 3);


        let each_space = |i: usize, j: usize| -> String {
            " ".repeat(max_space_each_col[j] - self.value[i][j].len() + 1)
        };

        let mut str = String::new();

        str += format!("┌{}┐\n", edge_space).as_str();

        for i in 0..self.dimension {
            str += "│";

            for j in 0..self.dimension {
                str += each_space(i, j).as_str();
                str += self.value[i][j].to_string().as_str();
            }

            str += ":";
            str += each_space(i, self.dimension).as_str();
            str += self.value[i][self.dimension].to_string().as_str();
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

    fn deno_1_fraction(value: Vec<isize>) -> Vec<Fraction> {
        value
            .into_iter()
            .map(|x| Fraction::new(x, 1))
            .collect()
    }

    fn gauss_temp(
        value: Vec<isize>,
        solution: Vec<isize>,
    ) {
        let mut matrix = Matrix::new(
            &deno_1_fraction(value)
        );
        matrix.gaussian_elimination(false);

        let solution = deno_1_fraction(solution);

        assert_eq!(matrix.extract(), solution);
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
        attempts: usize,
    ) {
        let mut matrix = Matrix::new(
            &deno_1_fraction(value)
        );
        let ans = matrix.jacobi_iterative(attempts, false);

        assert_eq!(ans, solution);
    }

    #[test]
    fn jakobi_four() {
        jakobi_temp(
            vec![
                10,  1, -1,  1,  55,
                 2,  9, -2,  4, -76,
                -1, -3, 12,  1,  75,
                -1,  2, -1, 15, -58
            ],
            vec![7, -8, 5, -2],
            50
        );
    }
}