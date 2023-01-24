use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::ops;

use ptree::style::Style;
use ptree::TreeItem;

pub use crate::ops::*;

#[derive(Debug, PartialEq, Clone)]
pub enum ScalarOp {
    None,
    Add,
    Mul,
    Tanh,
}

impl fmt::Display for ScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarOp::None => write!(f, ""),
            ScalarOp::Add => write!(f, "+"),
            ScalarOp::Mul => write!(f, "*"),
            ScalarOp::Tanh => write!(f, "tanh"),
        }
    }
}

#[derive(Debug, Clone)]
struct Scalar<'a> {
    val: f32,
    grad: RefCell<f32>,
    lhs_parent: Option<&'a Scalar<'a>>,
    rhs_parent: Option<&'a Scalar<'a>>,
    op: ScalarOp,
}

impl<'a> Scalar<'a> {
    fn new(val: f32) -> Scalar<'a> {
        Scalar {
            val,
            grad: RefCell::new(0.0),
            lhs_parent: None,
            rhs_parent: None,
            op: ScalarOp::None,
        }
    }

    fn new_with_parents<'b>(
        val: f32,
        lhs_parent: Option<&'b Scalar>,
        rhs_parent: Option<&'b Scalar>,
        op: ScalarOp,
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
        match self.op {
            ScalarOp::Add => {
                let lhs = self.lhs_parent.unwrap();
                let rhs = self.rhs_parent.unwrap();

                let old_lgrad = lhs.grad.borrow().clone();
                *lhs.grad.borrow_mut() = old_lgrad + *self.grad.borrow();

                let old_rgrad = rhs.grad.borrow().clone();
                *rhs.grad.borrow_mut() = old_rgrad + *self.grad.borrow();
            }
            ScalarOp::Mul => {
                let lhs = self.lhs_parent.unwrap();
                let rhs = self.rhs_parent.unwrap();

                let old_lgrad = lhs.grad.borrow().clone();
                *lhs.grad.borrow_mut() = old_lgrad + *self.grad.borrow() * rhs.val;

                let old_rgrad = rhs.grad.borrow().clone();
                *rhs.grad.borrow_mut() = old_rgrad + *self.grad.borrow() * lhs.val;
            }
            ScalarOp::Tanh => {
                let lhs = self.lhs_parent.unwrap();

                let old_lgrad = lhs.grad.borrow().clone();

                *lhs.grad.borrow_mut() = old_lgrad + (1.0 - self.val * self.val) * (*self.grad.borrow());
            }
            _ => {}
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
        let out = Scalar::new_with_parents(self.val + rhs.val, Some(self), Some(rhs), ScalarOp::Add);

        out
    }
}

impl<'a> ops::Add<f32> for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn add(self, rhs: f32) -> Self::Output {
        Scalar::new_with_parents(self.val + rhs, Some(self), None, ScalarOp::Add)
    }
}

impl<'a> ops::Add<&'a Scalar<'a>> for f32 {
    type Output = Scalar<'a>;

    fn add(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self + rhs.val, None, Some(rhs), ScalarOp::Add)
    }
}

impl<'a> ops::Mul for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn mul(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self.val * rhs.val, Some(self), Some(rhs), ScalarOp::Mul)
    }
}

impl<'a> ops::Mul<f32> for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn mul(self, rhs: f32) -> Self::Output {
        Scalar::new_with_parents(self.val * rhs, Some(self), None, ScalarOp::Mul)
    }
}

impl<'a> ops::Mul<&'a Scalar<'a>> for f32 {
    type Output = Scalar<'a>;

    fn mul(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self * rhs.val, None, Some(rhs), ScalarOp::Mul)
    }
}

impl<'a> Scalar<'a> {
    fn tanh(&'a self) -> Scalar<'a> {
        Scalar::new_with_parents(self.val.tanh(), Some(self), None, ScalarOp::Tanh)
    }
}

impl fmt::Display for Scalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.op != ScalarOp::None {
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

    let x1 = Scalar::new(2.0);
    let x2 = Scalar::new(0.0);
    
    let w1 = Scalar::new(-3.0);
    let w2 = Scalar::new(1.0);

    let b = Scalar::new(6.8813735870195432);

    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;
    let x1w1x2w2 = &x1w1 + &x2w2;

    let n = &x1w1x2w2 + &b;
    let o = n.tanh();

    *o.grad.borrow_mut() = 1.0;

    o.calc_grad();
    n.calc_grad();
    x1w1x2w2.calc_grad();
    x1w1.calc_grad();
    x2w2.calc_grad();
    b.calc_grad();
    w2.calc_grad();
    w1.calc_grad();
    x2.calc_grad();
    x1.calc_grad();

    ptree::print_tree(&&o).expect("Print tree error!");

}
