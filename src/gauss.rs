use std::fmt;
pub mod fraction;
use fraction::Fraction;

#[derive(Debug, Clone)]
pub struct Gauss {
    dimension: usize,
    value: Vec<Vec<Fraction>>,
}

impl Gauss {
    /// Creates a new Gauss object from a vector of fractions
    ///
    /// # Examples
    /// ```
    /// use gauss::gauss::Gauss;
    /// use gauss::fraction::Fraction;
    ///
    /// let gauss = Gauss::new(
    ///     vec![
    ///         (1, 1), ( 1, 1), (5, 1),
    ///         (2, 1), (-1, 1), (1, 1)
    ///     ]
    ///     .map(Fraction::new)
    ///     .collect()
    /// );
    /// ```
    pub const fn new(fractions: &[Fraction]) -> Self {

        // calculates the dimension of the matrix from the number of elements
        //
        // i.pow(n): number of elements on the left  side
        // i:        number of elements on the right side
        //
        // # Examples
        //
        // [ 1  1  1 |  6 ]
        // [ 2 -1  3 |  7 ]
        // [ 1  4 -1 | 10 ]
        //
        // elements_len = 12
        //
        // i = 0; i.pow(2) + i = 0  (Greater than 0)
        // i = 1; i.pow(2) + i = 2  (Greater than 0)
        // i = 2; i.pow(2) + i = 6  (Greater than 0)
        // i = 3; i.pow(2) + i = 12 (Equal to 0)
        //
        // dimension = 3
        let dimension = {
            let elements_len = fractions.len();
            let mut i: usize = 0;

            loop {
                let lefted_elements = elements_len - i.pow(2) - i;

                if lefted_elements == 0 {
                    // 
                    break i;
                } else if lefted_elements > 0 {
                    i += 1;
                } else {
                    panic!("Invalid input: not a valid dimension");
                }
            }
        };

        let value = fractions
            .chunks(dimension + 1)
            .map(|x| x.to_vec())
            .collect();

        
        Self { dimension, value }
    }

    pub fn solve(&mut self, print: bool) {
        for i in 0..self.dimension {
            self.divide_and_subtract(i);
            if print {
                println!("{}", self);
            }
        }
    }

    fn divide_and_subtract(&mut self, num: usize) {
        let pivot = self.value[num][num];
        let non_pivot_rows: Vec<usize>
            = (0..self.dimension)
                .filter(|&i| i != num)
                .collect();

        self.value[num]
            .iter_mut()
            .for_each(|x| *x /= pivot);

        for i in non_pivot_rows {
            let factor = self.value[i][num];

            for j in 0..self.dimension+1 {
                let value_num = self.value[num][j];
                self.value[i][j] -= factor * value_num;
            }
        }
    }

    fn extract(&self) -> Vec<isize> {
        let mut result = vec![];

        for i in 0..self.dimension {
            result.push(self.value[i][self.dimension]);
        }

        result
            .into_iter()
            .map(|x| x.numerator)
            .collect()
    }
}


impl fmt::Display for Gauss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut max_space_each_row = [0].repeat(self.dimension + 1);

        for i in 0..self.dimension + 1 {
            for j in 0..self.dimension {
                let len = self.value[j][i].len();
                if max_space_each_row[i] < len {
                    max_space_each_row[i] = len;
                }
            }
        }

        let mut str = String::new();
        let edge_space = " ".repeat(max_space_each_row.iter().sum::<usize>() + (self.dimension * 2) + 1);

        let each_space = |i: usize, j: usize| -> String {
            " ".repeat(max_space_each_row[j] - self.value[i][j].len() + 1)
        };

        str += format!("┌{}┐\n", edge_space).as_str();


        for i in 0..self.dimension {
            str += "│";

            for j in 0..self.dimension {
                str += each_space(i, j).as_str();
                str += self.value[i][j].to_string().as_str();
            }

            str += format!(
                ":{}{} │\n",
                each_space(i, self.dimension),
                self.value[i][self.dimension].to_string()
            ).as_str();
        }
        str += format!("└{}┘", edge_space).as_str();
        write!(f, "{}", str)
    }
}

// TESTS

#[cfg(test)]
mod tests {
    use super::*;

    fn test_temp(
        value: Vec<isize>,
        solution: Vec<isize>,
    ) {
        let mut gauss = Gauss::new(
            value
                .into_iter()
                .map(|x| Fraction::new(x, 1))
                .collect()
        );
        gauss.solve(false);

        assert_eq!(gauss.extract(), solution);
    }

    #[test]
    fn two() {
        test_temp(
            vec![
                1,  1, 5,
                2, -1, 1
            ],
            vec![2, 3]
        );
    }

    #[test]
    fn three() {
        test_temp(
            vec![
                1,  1,  1,  6,
                2, -1,  3,  7,
                1,  4, -1, 10,
            ],
            vec![3, 2, 1]
        );
    }

    #[test]
    fn four() {
        test_temp(
            vec![
                1,  1,  1,  1, 10,
                2,  3, -1,  1,  9,
                1, -1,  2,  3, 17,
                4,  1,  1, -2,  1
            ],
            vec![1, 2, 3, 4]
        );
    }
}