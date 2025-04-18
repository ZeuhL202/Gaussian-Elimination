use::std::fmt;
mod fraction;
use fraction::Fraction;

pub struct Gauss([[Fraction; 4]; 3]);

impl Gauss {
    pub fn new(a: [String; 12]) -> Self {
        let fractions = a.map(Fraction::new);
        let mut gauss = [[Fraction::new("0/1".to_string()); 4]; 3];

        for i in 0..3 {
            for j in 0..4 {
                gauss[i][j] = fractions[i * 4 + j].clone();
            }
        }
        Self(gauss)
    }

    pub fn solve(&mut self) {
        for i in 0..self.0.len() {
            self.divide_and_subtract(i);
        }
    }

    fn divide_and_subtract(&mut self, num: usize) {
        let pivot = self.0[num][num].clone();
        let non_pivot_rows: Vec<usize> = (0..self.0.len()).filter(|&i| i != num).collect();

        self.0[num].iter_mut().for_each(|x| *x /= pivot.clone());

        for i in non_pivot_rows {
            let factor = self.0[i][num].clone();

            for j in 0..self.0[i].len() {
                self.0[i][j] -= factor * self.0[num][j];
            }
        }
    }
}


impl fmt::Display for Gauss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut max_space_each_row: [usize; 4] = [0, 0, 0, 0];

        for i in 0..4 {
            for j in 0..3 {
                let len = self.0[j][i].len();
                if max_space_each_row[i] < len {
                    max_space_each_row[i] = len;
                }
            }
        }

        let mut str = String::new();
        let edge_space = " ".repeat(max_space_each_row.iter().sum::<usize>() + 6);

        let each_space = |i: usize, j: usize| -> String {
            " ".repeat(max_space_each_row[j] - self.0[i][j].len() + 1)
        };

        str += format!("┌{}┐\n", edge_space).as_str();


        for i in 0..3 {
            str += "│";

            for j in 0..3 {
                str += format!("{}{}", each_space(i, j), self.0[i][j].to_string()).as_ref();
            }

            str += format!(": {}{}│\n", self.0[i][3].to_string(), each_space(i, 3)).as_str();
        }
        str += format!("└{}┘", edge_space).as_str();
        write!(f, "{}", str)
    }
}