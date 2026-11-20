"""Minimal integer-linear model emitter for the accompanying Gurobi formulation.
All arithmetic is signed 16-bit; binary variables are declared as one bit.
"""
from dataclasses import dataclass

WIDTH = 16
class GRB:
    BINARY = 'B'
    INTEGER = 'I'
    CONTINUOUS = 'C'
    MAXIMIZE = -1

@dataclass(eq=False)
class Var:
    name: str
    width: int
    lb: int
    ub: int
    X: float = 0
    def expr(self): return Expr({self.name: 1}, 0)
    def __add__(self, other): return self.expr() + other
    def __radd__(self, other): return self.expr() + other
    def __sub__(self, other): return self.expr() - other
    def __rsub__(self, other): return as_expr(other) - self
    def __mul__(self, other): return self.expr() * other
    def __rmul__(self, other): return self.expr() * other
    def __eq__(self, other): return self.expr() == other
    def __le__(self, other): return self.expr() <= other
    def __ge__(self, other): return self.expr() >= other

class Expr:
    def __init__(self, terms=None, constant=0):
        self.terms = terms or {}
        self.constant = constant
    def __add__(self, other):
        other = as_expr(other); terms = self.terms.copy()
        for k, v in other.terms.items(): terms[k] = terms.get(k, 0) + v
        return Expr({k:v for k,v in terms.items() if v}, self.constant + other.constant)
    __radd__ = __add__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-as_expr(other))
    def __rsub__(self, other): return as_expr(other) - self
    def __mul__(self, n):
        if not isinstance(n, int): raise TypeError('integer coefficients only')
        return Expr({k:v*n for k,v in self.terms.items() if v*n}, self.constant*n)
    __rmul__ = __mul__
    def __eq__(self, other): return ('=', self - other)
    def __le__(self, other): return ('bvsle', self - other)
    def __ge__(self, other): return ('bvsge', self - other)

def as_expr(value):
    if isinstance(value, Expr): return value
    if isinstance(value, Var): return value.expr()
    if isinstance(value, int): return Expr(constant=value)
    raise TypeError(f'not an integer expression: {value!r}')

def LinExpr(value=0): return as_expr(value)
def quicksum(items):
    terms = {}; constant = 0
    for item in items:
        e = as_expr(item); constant += e.constant
        for k,v in e.terms.items(): terms[k] = terms.get(k,0)+v
    return Expr({k:v for k,v in terms.items() if v}, constant)

def bv(n, width=WIDTH): return f'(_ bv{n % (1 << width)} {width})'

class Model:
    def __init__(self, name):
        self.name = name; self.vars = {}; self.constraints = []; self.ModelSense = None
    def addVar(self, lb=0, ub=None, vtype=GRB.CONTINUOUS, name=''):
        if name in self.vars: raise ValueError(f'duplicate variable {name}')
        if vtype == GRB.BINARY: width, lb, ub = 1, 0, 1
        else:
            if ub is None or int(lb) != lb or int(ub) != ub or lb < 0:
                raise ValueError(f'finite nonnegative integer bounds required for {name}')
            lb, ub = int(lb), int(ub); width = max(1, ub.bit_length())
        var = Var(name, width, lb, ub); self.vars[name] = var
        if lb: self.addConstr(var >= lb)
        if ub != (1 << width) - 1: self.addConstr(var <= ub)
        return var
    def addConstr(self, constraint, name=''):
        if not isinstance(constraint, tuple): raise TypeError(f'invalid constraint {name}')
        self.constraints.append(constraint)
    def addGenConstrAnd(self, result, args, name=''):
        self.constraints.append(('booland', result, list(args)))
    def addGenConstrOr(self, result, args, name=''):
        self.constraints.append(('boolor', result, list(args)))
    def _term(self, name):
        v = self.vars[name]
        return name if v.width == WIDTH else f'((_ zero_extend {WIDTH-v.width}) {name})'
    def expression(self, expr):
        parts = []
        for name, coef in expr.terms.items():
            term = self._term(name)
            if coef != 1: term = f'(bvmul {bv(coef)} {term})'
            parts.append(term)
        if expr.constant or not parts: parts.append(bv(expr.constant))
        while len(parts) > 1:
            parts = [f'(bvadd {parts[i]} {parts[i+1]})' if i+1 < len(parts) else parts[i] for i in range(0,len(parts),2)]
        return parts[0]
    def assertion(self, c):
        op = c[0]
        if op in ('booland','boolor'):
            _, result, args = c
            operator = 'and' if op == 'booland' else 'or'
            rhs = f'({operator} ' + ' '.join(f'(= {a.name} #b1)' for a in args) + ')'
            return f'(assert (= (= {result.name} #b1) {rhs}))'
        return f'(assert ({op} {self.expression(c[1])} {bv(0)}))'
    def validate(self, extra=()):
        for v in self.vars.values():
            if not v.lb <= v.X <= v.ub or int(v.X) != v.X:
                raise ValueError(f"variable {v.name} outside its domain: {v.X}")
        for c in [*self.constraints, *extra]:
            op = c[0]
            if op in ("booland", "boolor"):
                _, result, args = c
                expected = all(a.X for a in args) if op == "booland" else any(a.X for a in args)
                valid = result.X == int(expected)
            else:
                expr = c[1]
                value = expr.constant + sum(coef * self.vars[name].X for name, coef in expr.terms.items())
                valid = (value == 0 if op == "=" else value <= 0 if op == "bvsle" else value >= 0)
            if not valid: raise ValueError(f"solver assignment violates {op} constraint")
    def smt(self, extra=(), values=False):
        lines = ['(set-logic QF_BV)', '(set-option :produce-models true)']
        lines += [f'(declare-fun {v.name} () (_ BitVec {v.width}))' for v in self.vars.values()]
        lines += [self.assertion(c) for c in self.constraints]
        lines += [self.assertion(c) for c in extra]
        lines.append('(check-sat)')
        if values: lines.append('(get-value (' + ' '.join(self.vars) + '))')
        lines.append('(exit)')
        return '\n'.join(lines) + '\n'
