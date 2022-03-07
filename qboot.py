import numpy
import scipy
import cvxpy
import numba
from numbers import Number
from collections.abc import Iterable

''' representation of coefficient '''
def _coef_repr(c):
    if c.imag == 0.:
        c = c.real
        if c == numpy.floor(c):
            if c == 1:
                txt = ' '
            elif c == -1:
                txt = '- '
            else:
                txt = '{:d} '.format(int(c))
        else: 
            txt = '{:.2f} '.format(c)
    elif c.real == 0.:
        c = c.imag
        if c == numpy.floor(c):
            if c == 1:
                txt = 'i '
            elif c == -1:
                txt = '-i '
            else:
                txt = '{:d}i '.format(int(c))
        else: 
            txt = '{:.2f}i '.format(c)
    else:
        txt = '({:.2f}) '.format(c).replace('j','i')
    return txt

''' Represents an element of an associative algebrra, which admits
    addition, multiplication (associative), scalar multiplication
    Parameters:
    terms - a dictionary of term:coef
        term - a hashable object expressing a basis element of the algebra
        coef - a number as the linear combination coefficient
'''
class Operator():
    def __init__(self, terms=None):
        self.terms = {} if terms is None else terms
        self.clear_cache()

    # clear cache variables (if any)
    # method will be called when the operator is modified in-place
    def clear_cache(self):
        self._H = None
        self._real = None
        self._imag = None
        
    def new(self, terms=None):
        return type(self)(terms)
    
    def copy(self):
        return self.new(self.terms.copy())

    def __len__(self):
        return len(self.terms)

    # on iterate, yield each term as a separate operator
    def __iter__(self):
        for term, coef in self.terms.items():
            yield self.new({term: coef})
 
    def __repr__(self):
        if self == 0:
            return '0'
        txt = ''
        for term in self.terms:
            txt_term = _coef_repr(self.terms[term])
            txt_term += self.term_repr(term)
            if txt != '' and txt_term[0] != '-':
                txt += '+'
            txt += txt_term
        return txt.strip()

    # provides a representation for operator term
    # (to be redefined by specific Operator subclasses)
    def term_repr(self, term):
        if len(term) == 0:
            return 'I'
        else:
            return repr(term)

    def __hash__(self):
        return hash(tuple(self.terms.items()))

    def __eq__(self, other):
        if other is 0:
            if len(self) != 0:
                for coef in self.terms.values():
                    if coef != 0:
                        return False
            return True
        elif isinstance(other, Number): 
            return self == other * one(self) # treat number as constant operator
        elif isinstance(other, Operator):
            return self.terms == other.terms # compare terms, using python dict eq (orderless)
        else:
            raise NotImplementedError("eq is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, Number): # scalar multiplication
            if other == 0:
                return zero(self)
            return self.new({term: coef * other for term, coef in self.terms.items()})
        elif isinstance(other, Operator): # operator dot product (trace product)
            return self.dot(other)
        elif isinstance(other, numpy.ndarray): # numpy array encountered
            return numpy.array(self) * other # promote self to numpy array
        else:
            raise NotImplementedError("mul is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
            
    def __rmul__(self, other):
        return self * other
    
    # scalar division
    def __truediv__(self, other):
        return self * (1/other)

    def __neg__(self):
        return self.new({term: -coef for term, coef in self.terms.items()})

    def __iadd__(self, other):
        if other == 0: # quick return on zero addition
            return self
        if isinstance(other, Number):
            other = other * one(self) # treat number as constant operator
        if not isinstance(other, Operator):
            raise NotImplementedError("add is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        for term in other.terms:
            if term in self.terms: # if term exists
                self.terms[term] += other.terms[term] # add coefficients
                if self.terms[term] == 0: # if cancelation happens 
                    self.terms.pop(term) # drop zero term
                    self.clear_cache() # as terms are updated, cache must be cleared
            else: # if term not exist
                coef = other.terms[term]
                if coef != 0: # if coefficient non-zero
                    self.terms[term] = coef # add term
                    self.clear_cache() # as terms are updated, cache must be cleared
        return self

    def __add__(self, other):
        if isinstance(other, numpy.ndarray):
            return other + self # hand over to numpy.array addition
        if other is None or other == 0: # quick return on zero addition
            return self
        if self == 0:
            return other
        new = self.copy()
        new += other
        return new

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    # operator multiplication
    def __matmul__(self, other):
        if not isinstance(other, Operator):
            raise NotImplementedError("matmul is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))            
        result = zero(self)
        for term_self in self.terms:
            for term_other in other.terms:
                coef = self.terms[term_self] * other.terms[term_other]
                result += coef * self.term_mul(term_self, term_other)
        return result

    # define multiplication of two basis operators in terms of their term experessions
    # (to be redefined by specific Operator subclasses)
    def term_mul(self, term1, term2):
        return self.new({term1+term2: 1})

    # compute commutator with other
    def commutate(self, other):
        return self @ other - other @ self

    # term signature (sign(A) = Tr(A A)/Tr(A^\dagger A))
    # (to be redefined by specific Operator subclasses)
    def term_sign(self, term):
        return 1

    # Hermitian conjugate
    def conjugate(self):
        return self.new({term: self.term_sign(term) * numpy.conjugate(coef) for term, coef in self.terms.items()})

    @property
    def H(self):
        if self._H is None:
            self._H = self.conjugate()
        return self._H

    @property
    def real(self):
        if self._real is None:
            self._real = ((self + self.H)/2)
        return self._real

    @property
    def imag(self):
        if self._imag is None:
            self._imag = ((self - self.H)/(2j))
        return self._imag

    @property
    def reim(self):
        return Operators([self.real, self.imag])

    # ordinary dot prooduct of two operators (without Hermitian conjugation)
    #   A.dot(B) = Tr(A B)
    # (assuming operator basis are othogonal, i.e. Tr(e_i e_j)=0 for i != j)
    def dot(self, other):
        if not isinstance(other, Operator):
            raise NotImplementedError("dot is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        if len(other) <= len(self):
            result = 0
            for term in other.terms:
                if term in self.terms:
                    result += self.term_sign(term) * self.terms[term] * other.terms[term]
            return result
        else:
            return other.dot(self)
 
    # Hermitian dot product of two operators (first operator gets Hermitian conjugated)
    #   A.vdot(B) = Tr(A^\dagger B)
    def vdot(self, other):
        if not isinstance(other, Operator):
            raise NotImplementedError("vdot is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        if len(other) <= len(self):
            result = 0
            for term in other.terms:
                if term in self.terms:
                    result += numpy.conjugate(self.terms[term]) * other.terms[term]
            return result
        else:
            return numpy.conjugate(other.vdot(self))

    # trace = the coefficient of the identity element
    def trace(self):
        return one(self).dot(self)

    # Hermitian norm of operator
    #   norm(A) = Tr(A^\dagger A)
    def norm(self):
        return numpy.sqrt(sum(abs(coef)**2 for coef in self.terms.values()))

    # normalize an operator
    def normalize(self, tol=1.e-10):
        norm = self.norm()
        if norm > tol:
            return self / norm
        else:
            return zero(self)

    # operator weight (weighted average of the weight function)
    # weigth function maps each term to a real number
    def weight(self, weight_func=len):
        p = numpy.array([abs(coef)**2 for coef in self.terms.values()])
        w = numpy.array([weight_func(term) for term in self.terms])
        psum = p.sum()
        if psum > 0:
            return w.dot(p) / psum
        else:
            return 0

    # round the coefficient to the given number of decimals
    def round(self, decimals=0):
        terms = {}
        for term, coef in self.terms.items():
            coef = numpy.round(coef, decimals=decimals)
            if coef != 0:
                terms[term] = coef
        return self.new(terms)

    # returns a iterator over generating operators
    def generators(self):
        elems = set()
        for term in self.terms:
            for elem in term:
                if elem not in elems:
                    elems.add(elem)
                    yield self.new({(elem,):1})

    # returns a iterator over basis operators
    def basis(self):
        terms = set()
        for term in self.terms:
            if term not in terms:
                terms.add(term)
                yield self.new({term:1}) 


# Universal operators
# the zero element of an associative algebra (identity of addition)
def zero(optype=None):
    if optype is None:
        optype = Operator
    elif isinstance(optype, Operator):
        optype = type(optype)
    return optype({})

# the unit element of an associative algebra (identity of multiplication)
def one(optype=None):
    if optype is None:
        optype = Operator
    elif isinstance(optype, Operator):
        optype = type(optype)
    return optype({(): 1})

opprod = numpy.frompyfunc((lambda x,y:x@y),2,1).reduce

''' Represents Majorana operator in Clifford algebra
    term: product of Clifford generators (labeled by their indices)
'''
class MajoranaOperator(Operator):
    def clear_cache(self):
        super(MajoranaOperator, self).clear_cache()
        self._parity = None
        self._loc_terms = None 

    # representation for operator term
    def term_repr(self, term):
        if len(term) == 0:
            return 'I '
        txt = ''
        for i in term:
            txt += 'χ{:d} '.format(i)
        return txt

    # term-level multiplication rule
    def term_mul(self, term1, term2):
        if len(term1) == 0:
            return self.new({term2: 1})
        if len(term2) == 0:
            return self.new({term1: 1})
        term1 = numpy.array(term1)
        term2 = numpy.array(term2)
        term, sign = maj_term_mul(term1, term2)
        term = tuple(term.tolist())
        return self.new({term: sign})

    # define term signature
    def term_sign(self, term):
        return 1-2*((len(term)//2)%2)

    # fermion parity (+1: even, -1: odd, 0: mixed)
    @property
    def parity(self):
        if self._parity is None:
            term_parity = numpy.array([len(term)%2 for term in self.terms])
            if numpy.all(term_parity == 0):
                self._parity = 1
            elif numpy.all(term_parity == 1):
                self._parity = -1
            else:
                self._parity = 0
        return self._parity

    # local term experessions
    @property
    def loc_terms(self):
        if self._loc_terms is None:
            self._loc_terms = {}
            for term in self.terms:
                for i in term:
                    if i in self._loc_terms:
                        self._loc_terms[i].add(term)
                    else:
                        self._loc_terms[i] = {term}
        return self._loc_terms

    # compute commutator with other
    def commutate(self, other):
        # commutator is localizable if either operator is even parity
        if self.parity == 1 or other.parity == 1:
            if len(other) <= len(self):
                result = zero(self)
                # single loop through terms
                for term_other in other.terms:
                    self_terms = set()
                    for i in term_other:
                        self_terms |= self.loc_terms.get(i, set())
                    for term_self in self_terms:
                        coef = self.terms[term_self] * other.terms[term_other]
                        result += coef * self.term_comm(term_self, term_other)
                return result
            else:
                return -other.commutate(self)
        else: # fall back to double loop
            result = zero(self)
            for term_self in self.terms:
                for term_other in other.terms:
                    coef = self.terms[term_self] * other.terms[term_other]
                    result += coef * self.term_comm(term_self, term_other)
            return result

    # term-level commutator
    def term_comm(self, term1, term2):
        if len(term1) == 0:
            return zero(self)
        if len(term2) == 0:
            return zero(self)
        term1 = numpy.array(term1)
        term2 = numpy.array(term2)
        term, sign = maj_term_comm(term1, term2)
        term = tuple(term.tolist())
        if sign == 0:
            return zero(self)
        else:
            return self.new({term: sign})


''' multiply two Majorana operator products 
    Input:
    term1, term2 - indices of Majorana products to merge
    (use numpy.array to avoid dispatch for different sizes)

    Output:
    term - indices of the merged Majorana product
    sign - permutation sign generated during the merge
'''
@numba.njit
def maj_term_mul(term1, term2):
    n1 = term1.size
    n2 = term2.size
    term = numpy.empty(n1 + n2, dtype=term1.dtype)
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    i  = 0 # term pointer
    ex = 0
    while i1 < n1 and i2 < n2:
        ind1 = term1[i1]
        ind2 = term2[i2]
        if ind1 == ind2:
            ex += n1 - i1 - 1
            i1 += 1
            i2 += 1
        else:
            if ind1 < ind2:
                term[i] = ind1
                i += 1
                i1 += 1
            else:
                ex += n1 - i1
                term[i] = ind2
                i += 1
                i2 += 1
    if i1 < n1:
        i_top = i + n1 - i1
        term[i:i_top] = term1[i1:n1]
        i = i_top
    if i2 < n2:
        i_top = i + n2 - i2
        term[i:i_top] = term2[i2:n2]
        i = i_top
    term = term[0:i]
    sign = 1-2*(ex%2)
    return term, sign 

''' commute two Majorana operator products 
    Input:
    term1, term2 - indices of Majorana products to merge
    (use numpy.array to avoid dispatch for different sizes)

    Output:
    term - indices of the resulting Majorana product
    sign - differential permutation sign generated during the merge
'''
@numba.njit
def maj_term_comm(term1, term2):
    n1 = term1.size
    n2 = term2.size
    term = numpy.empty(n1 + n2, dtype=term1.dtype)
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    i  = 0 # term pointer
    ex1 = 0
    ex2 = 0
    while i1 < n1 and i2 < n2:
        ind1 = term1[i1]
        ind2 = term2[i2]
        if ind1 == ind2:
            ex1 += n1 - i1 - 1
            ex2 += n2 - i2 - 1
            i1 += 1
            i2 += 1
        else:
            if ind1 < ind2:
                ex2 += n2 - i2
                term[i] = ind1
                i += 1
                i1 += 1
            else:
                ex1 += n1 - i1
                term[i] = ind2
                i += 1
                i2 += 1
    if i1 < n1:
        i_top = i + n1 - i1
        term[i:i_top] = term1[i1:n1]
        i = i_top
    if i2 < n2:
        i_top = i + n2 - i2
        term[i:i_top] = term2[i2:n2]
        i = i_top
    term = term[0:i]
    sign1 = 1-2*(ex1%2)
    sign2= 1-2*(ex2%2)
    return term, sign1-sign2 

''' Constructor for Majorana operator '''
# bare version
def maj(*args):
    if len(args) != 1:
        return maj(args)
    else:
        term = args[0]
        if isinstance(term, Number):
            return maj((int(term),))
        elif isinstance(term, tuple):
            return MajoranaOperator({term: 1})
        elif isinstance(term, (list, Iterable)):
            return maj(tuple(term))
        elif isinstance(term, numpy.ndarray):
            return maj(tuple(term.tolist()))
        else:
            raise NotImplementedError("majorana constructor is not implemented for '{}'".format(type(term).__name__))


''' Represents Pauli operator in Pauli algebra
    term: product of Pauli operators (labeled by (index, operator) pairs)
'''
class PauliOperator(Operator):
    def clear_cache(self):
        super(PauliOperator, self).clear_cache()
        self._loc_terms = None 

    # representation of operator term
    def term_repr(self, term):
        opnames = ('I','X','Y','Z')
        if len(term) == 0:
            return 'I'
        txt = ''
        for i, a in term:
            txt += opnames[a] + '{:d} '.format(i)
        return txt

    # term-level multiplication rule
    def term_mul(self, term1, term2):
        if len(term1) == 0:
            return self.new({term2: 1})
        if len(term2) == 0:
            return self.new({term1: 1})
        term1 = numpy.array(term1)
        term2 = numpy.array(term2)
        term, ipow = pauli_term_mul(term1, term2)
        term = tuple(tuple(pair) for pair in term.tolist())
        return self.new({term: 1j**ipow})

    # local term experessions
    @property
    def loc_terms(self):
        if self._loc_terms is None:
            self._loc_terms = {}
            for term in self.terms:
                for i, a in term:
                    if i in self._loc_terms:
                        self._loc_terms[i].add(term)
                    else:
                        self._loc_terms[i] = {term}
        return self._loc_terms

    # compute commutator with other
    def commutate(self, other):
        if len(other) <= len(self):
            result = zero(self)
            # single loop through terms
            for term_other in other.terms:
                self_terms = set()
                for i, _ in term_other:
                    self_terms |= self.loc_terms.get(i, set())
                for term_self in self_terms:
                    coef = self.terms[term_self] * other.terms[term_other]
                    result += coef * self.term_comm(term_self, term_other)
            return result
        else:
            return -other.commutate(self)
        
    # term-level commutator
    def term_comm(self, term1, term2):
        if len(term1) == 0:
            return zero(self)
        if len(term2) == 0:
            return zero(self)
        term1 = numpy.array(term1)
        term2 = numpy.array(term2)
        term, ipow = pauli_term_mul(term1, term2)
        term = tuple(tuple(pair) for pair in term.tolist())
        if ipow % 2 == 0:
            return zero(self)
        else:
            return self.new({term: 2*(1j**ipow)})

''' multiply two Pauli operator products 
    Input:
    term1, term2 - index-operator pairs of Pauli products to merge
    (use numpy.array to avoid dispatch for different sizes)

    Output:
    term - index-operator pairs of the merged Majorana product
    ipow - phase factor (in terms of power of i) generated during the merge
'''
@numba.njit
def pauli_term_mul(term1, term2):
    n1 = term1.shape[0]
    n2 = term2.shape[0]
    term = numpy.empty((n1 + n2, 2), dtype=term1.dtype)
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    i  = 0 # term pointer
    ipow = 0
    mu_table = numpy.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]])
    ipow_table = numpy.array([[0,0,0,0],[0,0,1,-1],[0,-1,0,1],[0,1,-1,0]])
    while i1 < n1 and i2 < n2:
        pos1 = term1[i1, 0]
        mu1  = term1[i1, 1]
        pos2 = term2[i2, 0]
        mu2  = term2[i2, 1]
        if pos1 == pos2:
            mu = mu_table[mu1, mu2]
            if mu != 0:
                ipow += ipow_table[mu1, mu2]
                term[i, 0] = pos1
                term[i, 1] = mu
                i += 1
            i1 += 1
            i2 += 1
        else:
            if pos1 < pos2:
                term[i, 0] = pos1
                term[i, 1] = mu1
                i += 1
                i1 += 1
            else:
                term[i, 0] = pos2
                term[i, 1] = mu2
                i += 1
                i2 += 1
    if i1 < n1:
        i_top = i + n1 - i1
        term[i:i_top] = term1[i1:n1]
        i = i_top
    if i2 < n2:
        i_top = i + n2 - i2
        term[i:i_top] = term2[i2:n2]
        i = i_top
    term = term[0:i]
    return term, ipow % 4

''' Constructor for Pauli operator '''
def pauli(*args):
    # reduce arguments
    if len(args) != 1:
        return pauli(args)
    else:
        obj = args[0]
        if isinstance(obj, Number):
            return pauli((int(obj),))
        elif isinstance(obj, str):
            return pauli(list(obj))
        elif isinstance(obj, dict):
            return pauli(obj.items())
        elif isinstance(obj, (tuple, list, numpy.ndarray, Iterable)):
            pass
        else:
            raise NotImplementedError("pauli constructor is not implemented for '{}'".format(type(obj).__name__))
    # start construction
    term = {}
    coef = 1
    i = 0
    a = 0
    opname = {'I':0,'X':1,'Y':2,'Z':3}
    itxt = ''
    state = 'head'
    # call to put down the current (i,a) pair
    def term_append(i, a):
        # if operator not trivial and position do not collide
        if a in [1, 2, 3] and i not in term: 
            term[i] = a
        return i + 1 # position shift forward
    # call to interpret itxt -> position
    def itxt_interp(itxt):
        i = int(itxt) # interpret itxt as a position
        itxt = '' # clear itxt
        state = 'body' # exit itxt state
        return i, itxt, state
    # iterate through items in the object
    for item in obj:
        if isinstance(item, tuple):
            if state == 'head':
                state = 'body' # enter body state
            elif state == 'body': # encouter a tuple in body state
                i = term_append(i, a) # put down the current (i,a) pair
            elif state == 'itxt': # encouter a tuple in itxt state
                i, itxt, state = itxt_interp(itxt) # interpret itxt -> position
                i = term_append(i, a) # put down the current (i,a) pair
            if len(item) == 2:
                if isinstance(item[0], str):
                    a = opname.get(item[0], None) # try get operator name
                    if isinstance(item[1], Number):
                        # item is of the form ('Z',1)
                        i = item[1] # record position
                elif isinstance(item[0], Number):
                    i, a = item
                    if isinstance(a, str):
                        # item is of the form (1,'Z')
                        a = opname.get(a, None) # try get operator name
                    elif isinstance(a, Number):
                        # item is of the form (1, 3)
                        pass
                    else:
                        a = None
        elif isinstance(item, str):
            # first check and interpret head decorators
            if item == '+':
                pass
            elif item == '-':
                if state == 'head': # effective if in head state
                    coef = -coef
            elif item == 'i':
                if state == 'head': # effective if in head state
                    coef *= 1j
            elif item in opname:
                if state == 'head': # encounter operator name in head state
                    state = 'body' # enter body state
                elif state == 'body': # encounter operator name in body state
                    i = term_append(i, a) # put down the current (i,a) pair
                elif state == 'itxt': # encounter operator name in itxt state
                    i, itxt, state = itxt_interp(itxt) # interpret itxt -> position
                    i = term_append(i, a) # put down the current (i,a) pair
                a = opname[item] # record operator name
            elif item.isnumeric():
                if state == 'head': # encounter numeric str in head state
                    state = 'body' # enter body state
                if state =='body': # encounter numeric str in body state
                    state = 'itxt' # enter itxt state
                itxt += item # record numeric str in itxt
            else: # encounter any other str
                if state == 'itxt': # if in the itxt state
                    i, itxt, state = itxt_interp(itxt) # interpret and end itxt state
        elif isinstance(item, Number):
            if state == 'head': # encounter a number in the head state
                a = item # interpret as operator name 
                i = term_append(i, a) # put down the current (i,a) pair
            elif state == 'body': # encounter a number in body state
                i = item # record as position
            elif state == 'itxt':
                itxt += str(item)
    if state == 'body': # reach end in body state
        i = term_append(i, a) # put down the current (i,a) pair
    elif state == 'itxt': # reach end in itxt state
        i, itxt, state = itxt_interp(itxt) # interpret itxt -> position
        i = term_append(i, a) # put down the current (i,a) pair
    # convert to tuple, sorted by position
    term = tuple((i, term[i]) for i in sorted(term))
    return PauliOperator({term: coef})

''' Represent an array of operators (subclass form numpy.ndarray)
    Parameter:
    ops -  an array of operators (each element is an Operator object)
        (to be parsed by numpy array constructor)
'''
class Operators(numpy.ndarray):
    def __new__(cls, ops):
        if isinstance(ops, Iterable):
            ops = list(ops)
        return numpy.asarray(ops).view(cls).cast()

    # automatic fall back to Operator if dimensionless
    def cast(self):
        if self.ndim == 0: # if Operators becomes dimensionless
            return self.item() # return item
        return self 

    def append(self, other):
        return type(self)(numpy.append(self, other))

    def __repr__(self, max_line_width=75):
        prefix = type(self).__name__ + '('
        suffix = ')'
        if self.size > 0 or self.shape == (0,):
            lst = numpy.array2string(self, max_line_width=max_line_width, 
                separator=', ', prefix=prefix, suffix=suffix)
        else: # show zero-length shape unless it is (0,)
            lst = "[], shape=%s" % (repr(arr.shape),)
        return prefix + lst + suffix

    # convert numerical result to numerical dtypes
    # caller must ensure data are numbers not operators
    def asarray(self):
        return numpy.real_if_close(numpy.asarray(self, dtype=numpy.complex))

    def __mul__(self, other):
        result = super(Operators, self).__mul__(other)
        if isinstance(other, (Operator, Operators)):
            result = result.asarray()
        return result

    def __matmul__(self, other):
        return super(Operators, self).__matmul__(other).cast()

    @property
    def H(self):
        return numpy.frompyfunc(lambda x: x.H,1,1)(self)

    @property
    def real(self):
        return numpy.frompyfunc(lambda x: x.real,1,1)(self)

    @property
    def imag(self):
        return numpy.frompyfunc(lambda x: x.imag,1,1)(self)

    @property
    def reim(self):
        return Operators(numpy.stack([self.real, self.imag], axis=-1))

    def trace(self):
        return numpy.frompyfunc(lambda x: x.trace(),1,1)(self).asarray()

    def norm(self):
        return numpy.frompyfunc(lambda x: x.norm(),1,1)(self).asarray()

    def normalize(self, tol=1.e-10):
        return numpy.frompyfunc(lambda x: x.normalize(tol),1,1)(self)

    def weight(self, weight_func=len):
        return numpy.frompyfunc(lambda x: x.weight(weight_func),1,1)(self).asarray()

    def round(self, decimals=0):
        return numpy.frompyfunc(lambda x: x.round(decimals),1,1)(self)

    # returns a iterator over generating operators
    def generators(self):
        elems = set()
        for op in self.flat:
            for term in op.terms:
                for elem in term:
                    if elem not in elems:
                        elems.add(elem)
                        yield op.new({(elem,):1})

    # returns a iterator over basis operators
    def basis(self):
        terms = set()
        for op in self.flat:
            for term in op.terms:
                if term not in terms:
                    terms.add(term)
                    yield op.new({term:1})

    # orthonormalization
    def orth(self):
        ops = OperatorSpace(self.basis())
        vecs = ops.represent(Operators(self.flat), axis=0)
        vecs = scipy.linalg.orth(vecs)
        return ops.reconstruct(vecs, axis=0).view(OperatorSpace)

''' Represent a operator space spanned by a set of operator basis
    (subclass of Operators, listing the basis operators)
    (basis operators are assumed to be orthonormal)
'''
class OperatorSpace(Operators):
    # extend the operator space by other operators or space
    def extend(self, other, n=1):
        if isinstance(other, Operator):
            other = Operators([other])
        if not isinstance(other, Operators):
            raise NotImplementedError("extend is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        new = self
        for _ in range(n):
            new = new.ope(other).orth().solve(self)
            self = self.append(new)
        return self
    
    # represent operator or operator array in the operator space
    # axis - specify the axis in which the vector representation resides
    def represent(self, other, axis=-1):
        if isinstance(other, Operator):
            return self.H * other # ignore axis
        if not isinstance(other, Operators):
            raise NotImplementedError("represent is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        # assuming numpy array holds operators
        axis = axis % (other.ndim + 1)
        other_shape = list(other.shape)
        other_shape.insert(axis, 1)
        basis_shape = [1]*other.ndim
        basis_shape.insert(axis,-1)
        result = self.H.reshape(basis_shape) * other.reshape(other_shape)
        return result

    # reconstruct operator from its representation
    # axis - specify the axis in which the vector representation resides
    def reconstruct(self, other, axis=-1):
        if not isinstance(other, numpy.ndarray):
            raise NotImplementedError("reconstruct is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        if other.ndim == 1:
            return self.dot(other) # ignore axis
        return numpy.tensordot(self, other, axes=([0],[axis])).view(Operators)
        

    # operator product expansion
    # returns: multiplication table of basis operators with another set of operators
    # (by default, OPE the basis operators with itself)
    def ope(self, other=None):
        if other is None:
            other = self
        if isinstance(other, Operator):
            other = Operators([other])
        if not isinstance(other, Operators):
            raise NotImplementedError("ope is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        outer = numpy.frompyfunc(lambda op1, op2: op1 @ op2, 2, 1).outer
        return outer(self, other).view(Operators)

    # adjoint action by a Lie algebra generator g
    # return: i[g, x] for x in operator basis
    def adjoint(self, other):
        if not isinstance(other, Operator):
            raise NotImplementedError("adjoint is not implemented between '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        adj = numpy.frompyfunc(lambda op: 1j*other.commutate(op), 1, 1)
        return adj(self).view(Operators)


    # Solve null space problem using scipy.linalg.null_space
    # Input: eqs - list of operators that should vanish
    # Output: null_ops - null operator space in which equations are solved
    def solve(self, eqs):
        eqs = eqs[eqs != 0]
        eqmat = self.represent(eqs)
        null = scipy.linalg.null_space(eqmat)
        null_ops = self.reconstruct(null, axis=0)
        return null_ops.view(OperatorSpace)


# Gram–Schmidt orthogonalization for a set of operator basis
def orthonormalize(basis, start=None, end=None, tol=1e-10):
    if start is None:
        start = 0
    if end is None:
        end = basis.size
    basis = basis.flat # treat as flat array
    for i in range(start, end):
        v = basis[i]
        us = basis[:i] # take all previous basis (orthonormal)
        if us.size == 0: # no preceeding basis
            u = v
        else: # orthogonalize with preceeding basis
            u = v - us.dot(us.H * v)
        u_norm = u.norm() # residual norm
        if u_norm > tol: # residual effectively non-zero
            basis[i] = u / u_norm # normalize
        else:
            basis[i] = zero(u)
    return basis[basis != 0] # filter out non-zero basis


''' Solve SDP problem using cvxpy 
    Problem formulation:
        min b.x, s.t. C + A[i].x[i] >> 0
    
    Input:
    b - objective vector
    C - constraint matrix (background)
    As - constraint matrices
    
    Output: (val, x)
    val - optimal objective value
    x - optimal solution of variable 
'''
def SDP(b, C, As):
    n = b.shape[0] # determine variable size
    x = cvxpy.Variable(n) # set variable
    objective = b @ x # set objective
    # compute constraint matrix
    A = C
    for i in range(n):
        A += As[i] * x[i]
    constraints = [A >> 0] # set constraints
    # create the SDP problem
    problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    problem.solve() # solve the problem
    return problem.value, x.value
            




















       