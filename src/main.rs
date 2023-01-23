use std::borrow::Borrow;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::ops;

use ptree::style::Style;
use ptree::TreeItem;

#[derive(Debug, Clone)]
struct Scalar<'a> {
    val: f32,
    grad: RefCell<f32>,
    lhs_parent: Option<&'a Scalar<'a>>,
    rhs_parent: Option<&'a Scalar<'a>>,
    op: String,
}

impl<'a> Scalar<'a> {
    fn new(val: f32) -> Scalar<'a> {
        Scalar {
            val,
            grad: RefCell::new(0.0),
            lhs_parent: None,
            rhs_parent: None,
            op: "".to_string(),
        }
    }

    fn new_with_parents<'b>(
        val: f32,
        lhs_parent: Option<&'b Scalar>,
        rhs_parent: Option<&'b Scalar>,
        op: String,
    ) -> Scalar<'b> {
        Scalar {
            val,
            grad: RefCell::new(0.0),
            lhs_parent,
            rhs_parent,
            op,
        }
    }

    fn calc_grad(&self) {
        if self.op == "+" {
            let lhs = self.lhs_parent.unwrap();
            let rhs = self.rhs_parent.unwrap();

            let old_lgrad = lhs.grad.borrow().clone();
            *lhs.grad.borrow_mut() = old_lgrad + *self.grad.borrow();

            let old_rgrad = rhs.grad.borrow().clone();
            *rhs.grad.borrow_mut() = old_rgrad + *self.grad.borrow();
        } else if self.op == "*" {
            let lhs = self.lhs_parent.unwrap();
            let rhs = self.rhs_parent.unwrap();

            let old_lgrad = lhs.grad.borrow().clone();
            *lhs.grad.borrow_mut() = old_lgrad + *self.grad.borrow() * rhs.val;

            let old_rgrad = rhs.grad.borrow().clone();
            *rhs.grad.borrow_mut() = old_rgrad + *self.grad.borrow() * lhs.val;
        } else {
            panic!("Unknown operation");
        }
    }
}

impl<'a> From<f32> for Scalar<'a> {
    fn from(value: f32) -> Self {
        Scalar::new(value)
    }
}

impl<'a> ops::Add for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn add(self, rhs: &'a Scalar) -> Self::Output {
        let out = Scalar::new_with_parents(self.val + rhs.val, Some(self), Some(rhs), "+".to_string());

        out
    }
}

impl<'a> ops::Add<f32> for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn add(self, rhs: f32) -> Self::Output {
        Scalar::new_with_parents(self.val + rhs, Some(self), None, "+".to_string())
    }
}

impl<'a> ops::Add<&'a Scalar<'a>> for f32 {
    type Output = Scalar<'a>;

    fn add(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self + rhs.val, None, Some(rhs), "+".to_string())
    }
}

impl<'a> ops::Mul for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn mul(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self.val * rhs.val, Some(self), Some(rhs), "+".to_string())
    }
}

impl<'a> ops::Mul<f32> for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn mul(self, rhs: f32) -> Self::Output {
        Scalar::new_with_parents(self.val * rhs, Some(self), None, "+".to_string())
    }
}

impl<'a> ops::Mul<&'a Scalar<'a>> for f32 {
    type Output = Scalar<'a>;

    fn mul(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self * rhs.val, None, Some(rhs), "+".to_string())
    }
}

impl<'a> Scalar<'a> {
    fn tanh(&'a self) -> Scalar<'a> {
        Scalar::new_with_parents(self.val.tanh(), Some(self), None, "tanh()".to_string())
    }
}

impl fmt::Display for Scalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.op != "" {
            write!(f, "Val({}; Δ{})<- {}", self.val, self.grad.borrow(), self.op)
        } else {
            write!(f, "Val({}; Δ{})", self.val, self.grad.borrow())
        }
    }
}

impl<'a> TreeItem for &Scalar<'a> {
    type Child = Self;

    fn write_self<W: io::Write>(&self, f: &mut W, style: &Style) -> io::Result<()> {
        write!(f, "{}", style.paint(self))
    }

    fn children(&self) -> Cow<[Self::Child]> {
        let mut childs = Vec::new();

        if let Some(lhs) = self.lhs_parent {
            childs.push(lhs);
        }
        if let Some(rhs) = self.rhs_parent {
            childs.push(rhs);
        }

        Cow::from(childs)
    }
}

fn main() {
    let a = Scalar::new(2.0);
    let b = Scalar::new(1.0);
    println!("a {}; b {}", a, b);

    let c = &a + &b;
    println!("c = &a + &b; c = {}", c);
    println!("&a + &b = {}", &a + &b);
    println!("{} + 5.0 = {}", a, &a + 5.0);
    println!("5.0 + {} = {}", a, 5.0 + &a);
    println!("&(&a + &b) + 5.0 = {}", &(&a + &b) + 5.0);

    let d = &c + 2.0;
    let e = Scalar::new(10.0);
    let f = &d * &e;
    let g = 0.01 * &f;
    let h = g.tanh();

    ptree::print_tree(&&h).expect("Print tree error!");
}
