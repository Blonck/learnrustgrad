use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::ops;

use ptree::style::Style;
use ptree::TreeItem;

#[derive(Debug, PartialEq, Clone)]
pub enum ScalarOp {
    None,
    Add,
    Div,
    Mul,
    Powi(i32),
    Tanh,
}

impl fmt::Display for ScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarOp::None => write!(f, ""),
            ScalarOp::Add => write!(f, "+"),
            ScalarOp::Mul => write!(f, "*"),
            ScalarOp::Powi(i) => write!(f, "powi({})", i),
            ScalarOp::Tanh => write!(f, "tanh"),
            ScalarOp::Div => write!(f, "/"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scalar<'a> {
    pub val: f32,
    pub grad: RefCell<f32>,
    lhs_parent: Option<&'a Scalar<'a>>,
    rhs_parent: Option<&'a Scalar<'a>>,
    op: ScalarOp,
}

impl<'a> Scalar<'a> {
    pub fn new(val: f32) -> Scalar<'a> {
        Scalar {
            val,
            grad: RefCell::new(0.0),
            lhs_parent: None,
            rhs_parent: None,
            op: ScalarOp::None,
        }
    }

    pub fn new_with_parents<'b>(
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

    pub fn calc_grad(&self) {
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
                *lhs.grad.borrow_mut() = old_lgrad + *self.grad.borrow() / rhs.val;

                let old_rgrad = rhs.grad.borrow().clone();
                *rhs.grad.borrow_mut() = old_rgrad + *self.grad.borrow() * lhs.val * rhs.val.powi(-2);
            }
            ScalarOp::Div => {
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
            ScalarOp::Powi(i) => {
                let lhs = self.lhs_parent.unwrap();
                let old_lgrad = lhs.grad.borrow().clone();

                *lhs.grad.borrow_mut() = old_lgrad + (lhs.val.powi(i - 1)) * (*self.grad.borrow());
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
    pub fn tanh(&'a self) -> Scalar<'a> {
        Scalar::new_with_parents(self.val.tanh(), Some(self), None, ScalarOp::Tanh)
    }
}

impl<'a> Scalar<'a> {
    pub fn powi(&'a self, n: i32) -> Scalar<'a> {
        Scalar::new_with_parents(self.val.powi(n), Some(self), None, ScalarOp::Powi(n))
    }
}

impl<'a> ops::Div for &'a Scalar<'a> {
    type Output = Scalar<'a>;

    fn div(self, rhs: &'a Scalar) -> Self::Output {
        Scalar::new_with_parents(self.val / rhs.val, Some(self), Some(rhs), ScalarOp::Div)
        // could use existing operators self * &rhs.powi(-1)
        // but &rhs.powi(-1) would be a dangling temporary variable
        // which also have to be returned to caller, e.g. some chained expression
        // ChainedResult<Scalar, Scalar> which could on dereferencing only return first element 
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
