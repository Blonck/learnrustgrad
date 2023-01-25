mod scalar;

use scalar::Scalar;

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
    let o = n.powi(2);
    let p = o.tanh();

    *p.grad.borrow_mut() = 1.0;

    p.calc_grad();
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
