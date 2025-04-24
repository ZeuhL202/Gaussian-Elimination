use::std::{fmt, ops};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Fraction {
    // The sign is represented by the numerator
    pub numerator: isize,
    // The denominator is always positive
    pub denominator: usize,
}

/// greatest common divisor
///
/// # Examples
/// ```
/// use gauss::fraction::gcd;
/// assert_eq!(gcd(12,  8), 4);
/// assert_eq!(gcd( 8, 12), 4);
/// assert_eq!(gcd( 7,  5), 1);
/// ```
const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        return a
    }
    gcd(b, a % b)
}

/// least common multiple
///
/// # Examples
/// ```
/// use gauss::fraction::lcm;
/// assert_eq!(lcm(12,  8), 24);
/// assert_eq!(lcm( 8, 12), 24);
/// assert_eq!(lcm( 7,  5), 35);
/// ```
const fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

/// 分数の構造体
impl Fraction {
    /// creates a new fraction
    pub const fn new(numerator: isize, denominator: usize) -> Self {
        Self {
            numerator: numerator,
            denominator: denominator,
        }
    }

    /// length of the fraction string
    ///
    /// # Examples
    /// ```
    /// use gauss::fraction::Fraction;
    ///
    /// assert_eq!(Fraction::new( 1, 2).len(), 3); //  "1/2" is 3 characters long
    /// assert_eq!(Fraction::new(-1, 3).len(), 4); // "-1/3" is 4 characters long
    /// assert_eq!(Fraction::new( 1, 1).len(), 1); //    "1" is 1 character  long
    /// ```
    pub const fn len(&self) -> usize {
        if self.denominator == 1 {
            return self.numerator.to_be_bytes().len()
        }
        self.numerator.to_be_bytes().len() + 1 + self.denominator.to_be_bytes().len()
    }

    /// reduces the fraction
    ///
    /// # Examples
    /// ```
    /// use gauss::fraction::Fraction;
    ///
    /// let mut fraction = Fraction::new(4, 8);
    /// fraction.reduce();
    /// assert_eq!(fraction, Fraction::new(1, 2));
    /// ```
    pub const fn reduce(&mut self) {
        let numerator = self.numerator.abs() as usize;

        // return if the fraction is already reduced
        if gcd(numerator, self.denominator) == 1 {
            return
        }

        let gcd = gcd(numerator, self.denominator);
        self.numerator   /= gcd as isize;
        self.denominator /= gcd;
    }
}

/// Match two fractions into their common denominator
///
/// # Examples
/// ```
/// use gauss::fraction::{Fraction, match_each_denominator};
/// let mut a = Fraction::new(1, 2);
/// let mut b = Fraction::new(3, 4);
/// match_each_denominator(&mut a, &mut b);
/// assert_eq!(a, Fraction::new(2, 4));
/// assert_eq!(b, Fraction::new(3, 4));
/// ```
pub const fn match_each_denominator(a: &mut Fraction, b: &mut Fraction) {

    // return if the denominators are the same
    if a.denominator == b.denominator {
        return
    }

    let lcm = lcm(a.denominator, b.denominator);
    a.numerator *= (lcm / a.denominator) as isize;
    b.numerator *= (lcm / b.denominator) as isize;
    a.denominator = lcm;
    b.denominator = lcm;
}

impl ops::Add for Fraction {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self {
        match_each_denominator(&mut self, &mut other);

        let numerator   = self.numerator + other.numerator;
        let denominator = self.denominator;

        let mut result = Self { numerator, denominator };
        result.reduce();
        result
    }
}

impl ops::AddAssign for Fraction {
    fn add_assign(&mut self, mut other: Self) {
        match_each_denominator(self, &mut other);

        self.numerator += other.numerator;
        self.reduce();
    }
}

impl ops::Sub for Fraction {
    type Output = Self;

    fn sub(mut self, mut other: Self) -> Self {
        match_each_denominator(&mut self, &mut other);

        let numerator   = self.numerator - other.numerator;
        let denominator = self.denominator;

        let mut result = Self { numerator, denominator };
        result.reduce();
        result
    }
}

impl ops::SubAssign for Fraction {
    fn sub_assign(&mut self, mut other: Self) {
        match_each_denominator(self, &mut other);

        self.numerator -= other.numerator;
        self.reduce();
    }
}

impl ops::Mul for Fraction {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let numerator   = self.numerator   * other.numerator;
        let denominator = self.denominator * other.denominator;

        let mut result = Self { numerator, denominator };
        result.reduce();
        result
    }
}

impl ops::MulAssign for Fraction {
    fn mul_assign(&mut self, other: Self) {
        self.numerator   *= other.numerator;
        self.denominator *= other.denominator;
        self.reduce();
    }
}

impl ops::Div for Fraction {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut numerator   = self.numerator.abs() * (other.denominator     as isize);
        let     denominator = self.denominator     * (other.numerator.abs() as usize);

        // if the signs are different, the result is negative
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
        self.numerator   *= other.denominator     as isize;
        self.denominator *= other.numerator.abs() as usize;

        // if the signs are different, the result is negative
        if self.numerator.is_negative() ^ other.numerator.is_negative() {
            self.numerator = -self.numerator;
        }

        self.reduce();
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        // if the denominator is 1, just print the numerator
        if self.denominator == 1 {
            return write!(f, "{}", self.numerator);
        }
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

// TESTS ARE HERE BELOW

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_positive() {
        let mut a = Fraction::new(1, 2);
        let     b = Fraction::new(3, 4);
        a += b;
        assert_eq!(a, Fraction::new(5, 4));
    }

    #[test]
    fn test_add_negative() {
        let mut a = Fraction::new(-1, 2);
        let     b = Fraction::new(-3, 4);
        a += b;
        assert_eq!(a, Fraction::new(-5, 4));
    }

    #[test]
    fn test_sub_positive() {
        let mut a = Fraction::new(1, 2);
        let     b = Fraction::new(3, 4);
        a -= b;
        assert_eq!(a, Fraction::new(-1, 4));
    }

    #[test]
    fn test_sub_negative() {
        let mut a = Fraction::new(-1, 2);
        let     b = Fraction::new(-3, 4);
        a -= b;
        assert_eq!(a, Fraction::new(1, 4));
    }

    #[test]
    fn test_mul_p_p() {
        let a = Fraction::new(1, 2);
        let b = Fraction::new(3, 4);
        let result = a * b;
        assert_eq!(result, Fraction::new(3, 8));
    }

    #[test]
    fn test_mul_p_n() {
        let a = Fraction::new( 1, 2);
        let b = Fraction::new(-3, 4);
        let result = a * b;
        assert_eq!(result, Fraction::new(-3, 8));
    }

    #[test]
    fn test_mul_n_p() {
        let a = Fraction::new(-1, 2);
        let b = Fraction::new( 3, 4);
        let result = a * b;
        assert_eq!(result, Fraction::new(-3, 8));
    }

    #[test]
    fn test_mul_n_n() {
        let a = Fraction::new(-1, 2);
        let b = Fraction::new(-3, 4);
        let result = a * b;
        assert_eq!(result, Fraction::new(3, 8));
    }

    #[test]
    fn test_div_p_p() {
        let a = Fraction::new(1, 2);
        let b = Fraction::new(3, 4);
        let result = a / b;
        assert_eq!(result, Fraction::new(2, 3));
    }

    #[test]
    fn test_div_p_n() {
        let a = Fraction::new( 1, 2);
        let b = Fraction::new(-3, 4);
        let result = a / b;
        assert_eq!(result, Fraction::new(-2, 3));
    }

    #[test]
    fn test_div_n_p() {
        let a = Fraction::new(-1, 2);
        let b = Fraction::new( 3, 4);
        let result = a / b;
        assert_eq!(result, Fraction::new(-2, 3));
    }

    #[test]
    fn test_div_n_n() {
        let a = Fraction::new(-1, 2);
        let b = Fraction::new(-3, 4);
        let result = a / b;
        assert_eq!(result, Fraction::new(2, 3));
    }

    #[test]
    fn test_match_each_denominator() {
        let mut a = Fraction::new(1, 2);
        let mut b = Fraction::new(3, 4);
        match_each_denominator(&mut a, &mut b);
        assert_eq!(a, Fraction::new(2, 4));
        assert_eq!(b, Fraction::new(3, 4));
    }

    #[test]
    fn test_reduce() {
        let mut a = Fraction::new(4, 8);
        a.reduce();
        assert_eq!(a, Fraction::new(1, 2));
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
