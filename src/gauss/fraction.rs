use::std::{fmt, ops};

/// 分数
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Fraction {
    /// 分子 (符号を管理するために isize)
    pub numerator: isize,
    /// 分母
    pub denominator: usize,
}

/// 最大公約数
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        return a
    }
    gcd(b, a % b)
}

/// 最小公倍数
fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

/// 分母が0のときにパニックを起こす
fn panic_denominator_zero(denominator: usize) {
    if denominator == 0 {
        panic!("denominator cannot be zero");
    }
}

/// 分数の構造体
impl Fraction {
    /// 分数を生成する
    pub fn new(format: String) -> Self {
        let (numerator, denominator) = format.split_once('/')
            .expect("Invalid format. Expected 'numerator/denominator'");

        let numerator = numerator.parse::<isize>()
            .expect("Invalid numerator format");

        let denominator = denominator.parse::<usize>()
            .expect("Invalid denominator format");

        panic_denominator_zero(denominator);

        Self {
            numerator: numerator,
            denominator: denominator,
        }
    }

    /// 分数の文字列の数
    pub fn len(&self) -> usize {
        if self.denominator == 1 {
            return self.numerator.to_string().len()
        }
        self.numerator.to_string().len() + 1 + self.denominator.to_string().len()
    }

    /// 約分する
    pub fn reduce(&mut self) {
        if gcd(self.numerator.abs() as usize, self.denominator) == 1 {
            return
        }
        let gcd = gcd(self.numerator.abs() as usize, self.denominator);
        self.numerator = self.numerator / (gcd as isize);
        self.denominator /= gcd;
    }
}

/// 2つの分数を通分する
pub fn align_each_denominator(a: &mut Fraction, b: &mut Fraction) {
    if a.denominator == b.denominator {
        return
    }
    let lcm = lcm(a.denominator, b.denominator);
    a.numerator *= (lcm / a.denominator) as isize;
    b.numerator *= (lcm / b.denominator) as isize;
    a.denominator = lcm;
    b.denominator = lcm;
}

/// 和
impl ops::Add for Fraction {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self {
        align_each_denominator(&mut self, &mut other);

        let numerator = self.numerator + other.numerator;
        let denominator = self.denominator;

        let mut result = Self {
            numerator,
            denominator,
        };
        result.reduce();
        result
    }
}

impl ops::AddAssign for Fraction {
    fn add_assign(&mut self, mut other: Self) {
        align_each_denominator(self, &mut other);

        self.numerator += other.numerator;
        self.reduce();
    }
}

/// 差
impl ops::Sub for Fraction {
    type Output = Self;

    fn sub(mut self, mut other: Self) -> Self {
        align_each_denominator(&mut self, &mut other);

        let numerator = self.numerator - other.numerator;
        let denominator = self.denominator;

        let mut result = Self {
            numerator,
            denominator,
        };
        result.reduce();
        result
    }
}

impl ops::SubAssign for Fraction {
    fn sub_assign(&mut self, mut other: Self) {
        align_each_denominator(self, &mut other);

        self.numerator -= other.numerator;
        self.reduce();
    }
}

/// 積
impl ops::Mul for Fraction {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let numerator = self.numerator * other.numerator;
        let denominator = self.denominator * other.denominator;

        let mut result = Self {
            numerator,
            denominator,
        };
        result.reduce();
        result
    }
}

impl ops::MulAssign for Fraction {
    fn mul_assign(&mut self, other: Self) {
        self.numerator *= other.numerator;
        self.denominator *= other.denominator;
        self.reduce();
    }
}

/// 商
impl ops::Div for Fraction {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut numerator = self.numerator.abs() * (other.denominator as isize);
        let denominator = self.denominator * (other.numerator.abs() as usize);

        if self.numerator.is_negative() ^ other.numerator.is_negative() {
            numerator = -numerator;
        }

        let mut result = Self {
            numerator,
            denominator,
        };
        result.reduce();
        result
    }
}

impl ops::DivAssign for Fraction {
    fn div_assign(&mut self, other: Self) {
        let mut  numerator = self.numerator.abs() * (other.denominator as isize);
        let denominator = self.denominator * (other.numerator.abs() as usize);

        if self.numerator.is_negative() ^ other.numerator.is_negative() {
            numerator = -numerator;
        }
        let mut result = Self {
            numerator,
            denominator,
        };
        result.reduce();
        self.numerator = result.numerator;
        self.denominator = result.denominator;
    }
}

/// 表示
impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            return write!(f, "{}", self.numerator);
        }
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

// TESTS

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_positive() {
        let mut a = Fraction::new("1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        a += b;
        assert_eq!(a, Fraction::new("5/4".to_string()));
    }

    #[test]
    fn test_add_negative() {
        let mut a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        a += b;
        assert_eq!(a, Fraction::new("-5/4".to_string()));
    }

    #[test]
    fn test_sub_positive() {
        let mut a = Fraction::new("1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        a -= b;
        assert_eq!(a, Fraction::new("-1/4".to_string()));
    }

    #[test]
    fn test_sub_negative() {
        let mut a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        a -= b;
        assert_eq!(a, Fraction::new("1/4".to_string()));
    }

    #[test]
    fn test_mul_p_p() {
        let a = Fraction::new("1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        let result = a * b;
        assert_eq!(result, Fraction::new("3/8".to_string()));
    }

    #[test]
    fn test_mul_p_n() {
        let a = Fraction::new("1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        let result = a * b;
        assert_eq!(result, Fraction::new("-3/8".to_string()));
    }

    #[test]
    fn test_mul_n_p() {
        let a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        let result = a * b;
        assert_eq!(result, Fraction::new("-3/8".to_string()));
    }

    #[test]
    fn test_mul_n_n() {
        let a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        let result = a * b;
        assert_eq!(result, Fraction::new("3/8".to_string()));
    }

    #[test]
    fn test_div_p_p() {
        let a = Fraction::new("1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        let result = a / b;
        assert_eq!(result, Fraction::new("2/3".to_string()));
    }

    #[test]
    fn test_div_p_n() {
        let a = Fraction::new("1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        let result = a / b;
        assert_eq!(result, Fraction::new("-2/3".to_string()));
    }

    #[test]
    fn test_div_n_p() {
        let a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("3/4".to_string());
        let result = a / b;
        assert_eq!(result, Fraction::new("-2/3".to_string()));
    }

    #[test]
    fn test_div_n_n() {
        let a = Fraction::new("-1/2".to_string());
        let b = Fraction::new("-3/4".to_string());
        let result = a / b;
        assert_eq!(result, Fraction::new("2/3".to_string()));
    }

    #[test]
    fn test_align_each_denominator() {
        let mut a = Fraction::new("1/2".to_string());
        let mut b = Fraction::new("3/4".to_string());
        align_each_denominator(&mut a, &mut b);
        assert_eq!(a, Fraction::new("2/4".to_string()));
        assert_eq!(b, Fraction::new("3/4".to_string()));
    }

    #[test]
    fn test_reduce() {
        let mut a = Fraction::new("4/8".to_string());
        a.reduce();
        assert_eq!(a, Fraction::new("1/2".to_string()));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 5), 1);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(12, 8), 24);
        assert_eq!(lcm(7, 5), 35);
    }
}
