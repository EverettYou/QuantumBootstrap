import math

class Operator():
    ''' Represents an element of an associative algebrra, 
        which admits addition, multiplication (associative), 
        scalar multiplication, trace and inner product

        Parameters:
        terms: dict - a dictionary of {term: coef, ...}
            term: tuple - spicification of a basis element (a term)
            coef: complex  - the linear combination coefficient 

        Attributes:
        tol: real - tolerance level in operator algebra,
                    coefficient smaller than tol will be treated
                    as zero. '''
    tol = 1.e-12
    def __init__(self, terms=None):
        self.terms = {} if terms is None else terms
        self._H = None    # cache Hermitian conjugate
        self._real = None # cache Hermitian part
        self._imag = None # cache anti-Hermitian part

    # ---- representation ----
    def __repr__(self):
        if self == 0: # zero operator
            return '0'
        # representation of coefficient
        def _coef_repr(c):
            if c.imag == 0.:
                c = c.real
                if c == round(c):
                    if c == 1:
                        txt = ' '
                    elif c == -1:
                        txt = '- '
                    else:
                        txt = f'{int(c):d} '
                else: 
                    txt = f'{c:.3g} '
            elif c.real == 0.:
                c = c.imag
                if c == round(c):
                    if c == 1:
                        txt = 'i '
                    elif c == -1:
                        txt = '-i '
                    else:
                        txt = f'{int(c):d}i '
                else: 
                    txt = f'{c:.3g}i '
            else:
                txt = f'({c:.3g}) '.replace('j','i')
            return txt
        txt = ''
        for term in self.terms:
            txt_term = _coef_repr(self.terms[term])
            txt_term += self.term_repr(term)
            if txt != '' and txt_term[0] != '-':
                txt += '+'
            txt += txt_term
        return txt.strip()

    # !!! to be redefined by specific Operator subclasses
    def term_repr(self, term):
        ''' provides a representation for operator term '''
        if len(term) == 0:
            return 'I'
        else:
            return repr(term)

    # ---- iterable ----
    def __iter__(self):
        ''' on iteration, yield each term as a separate operator '''
        for term, coef in self.terms.items():
            yield type(self)({term: coef})

    # ---- comparison ----
    def __eq__(self, other):
        ''' compare if self and other are the same operator 
            (self == other) '''
        if other == 0:
            return self.terms == {}
        elif isinstance(other, Operator):
            return self.terms == other.terms
        else:
            return (self - other).terms == {}

    # ---- linear algebra ----
    def __mul__(self, other):
        ''' scalar multiplication (A * x)
            Input:
            other: number - scalar number to multiply '''
        if other == 0:
            return zero(self)
        return type(self)({term: coef * other for term, coef in self.terms.items()})

    def __rmul__(self, other):
        ''' scalar multiplication (x * A)
            Input:
            other: number - scalar number to multiply '''
        return self * other

    def __imul__(self, other):
        ''' scalar multiplication (in-place) (A *= x)
            Input:
            other: number - scalar number to multiply '''
        if other == 0:
            self.terms = {}
        else:
            for term in self.terms:
                self.terms[term] *= other
        return self

    def __truediv__(self, other):
        ''' scalar division (A / x)
            Input:
            other: number - scalar number to devide '''
        return self * (1/other)

    def __itruediv__(self, other):
        ''' scalar division (in-place) (A /= x)
            Input:
            other: number - scalar number to devide '''
        for term in self.terms:
            self.terms[term] /= other
        return self

    def __neg__(self):
        ''' operator negation (- A) '''
        return type(self)({term: -coef for term, coef in self.terms.items()})

    def __add__(self, other):
        ''' operator addition (A + B)
            Input:
            other: Operator - the operator to add
                   number - treated as scalar multiple of identity '''
        if other is None:
            return self
        if not isinstance(other, Operator):
            other = unit(self) * other # non-operators are treated as numbers
        if len(other.terms) <= len(self.terms):
            shorter_terms, longer_terms = other.terms, self.terms.copy()
        else:
            shorter_terms, longer_terms = self.terms, other.terms.copy()
        for term in shorter_terms: # iterate through the shorter 
            if term in longer_terms: # lookup in the longer
                longer_terms[term] += shorter_terms[term]
                if abs(longer_terms[term]) < self.tol:
                    longer_terms.pop(term)
            else:
                coef = shorter_terms[term]
                if abs(coef) >= self.tol:
                    longer_terms[term] = coef
        return type(self)(longer_terms)

    def __radd__(self, other):
        ''' operator addition (B + A) 
            Input:
            other: Operator - the operator to add
                   number - treated as scalar multiple of identity '''
        return self + other

    def __iadd__(self, other):
        ''' operator addition (in-place) (A += B)
            Input:
            other: Operator - the operator to add
                   number - treated as scalar multiple of identity '''
        if other is None:
            return self
        if not isinstance(other, Operator):
            other = unit(self) * other # non-operators are treated as numbers
        for term in other.terms: # iterate through terms in other 
            if term in self.terms:
                self.terms[term] += other.terms[term]
                if abs(self.terms[term]) < self.tol:
                    self.terms.pop(term)
            else:
                coef = other.terms[term]
                if abs(coef) >= self.tol:
                    self.terms[term] = coef
        return self

    def __sub__(self, other):
        ''' operator subtraction (A - B)
            Input:
            other: Operator - the operator to add
                   number - treated as scalar multiple of identity '''
        return self + (-other)

    def __isub__(self, other):
        ''' operator subtraction (in-place) (A -= B)
            Input:
            other: Operator - the operator to add
                   number - treated as scalar multiple of identity '''
        if other is None:
            return self
        if not isinstance(other, Operator):
            other = unit(self) * other # non-operators are treated as numbers
        for term in other.terms: # iterate through terms in other 
            if term in self.terms:
                self.terms[term] -= other.terms[term]
                if abs(self.terms[term]) < self.tol:
                    self.terms.pop(term)
            else:
                coef = other.terms[term]
                if abs(coef) >= self.tol:
                    self.terms[term] = -coef
        return self

    # ---- monoidal algebra ----
    def __matmul__(self, other):
        ''' operator multiplication (A @ B)
            Input:
            other: Operator - the operator to mutiply '''
        result = zero(self)
        for term_self in self.terms:
            for term_other in other.terms:
                coef = self.terms[term_self] * other.terms[term_other]
                term_mul = self.term_mul(term_self, term_other)
                term_mul *= coef
                result += term_mul 
        return result

    # !!! to be redefined by specific Operator subclasses
    def term_mul(self, term1, term2):
        ''' define multiplication of two terms 
            Input:
            term1: tuple - first term to multiply
            term2: tuple - second term to multiply 
            Output:
            term1 @ term2: Operator '''
        return type(self)({term1+term2: 1})

    def commutate(self, other):
        ''' commutator of self with other 
            [A, B] = A @ B - B @ A
            Input:
            other: Operator - the operator to commutate with '''
        return self @ other - other @ self

    # ---- adjoint and inner product structure ----
    def adjoint(self):
        return type(self)({term: self.term_adj_sign(term) * coef.conjugate() for term, coef in self.terms.items()})

    # !!! to be redefined by specific Operator subclasses
    def term_adj_sign(self, term):
        ''' term adjoint sign 
            the additional sign generated when taking the adjoint
            of a term (this happens when the term is anti-Hermitian)
            Input:
            term: tuple - the term in consideration '''
        return 1

    @property
    def H(self):
        ''' Hermitian conjugation '''
        if self._H is None:
            self._H = self.adjoint()
        return self._H

    @property
    def real(self):
        ''' Hermitian part '''
        if self._real is None:
            self._real = ((self + self.H)/2)
        return self._real

    @property
    def imag(self):
        ''' anti-Hermitian part '''
        if self._imag is None:
            self._imag = ((self - self.H)/(2j))
        return self._imag

    @property
    def reim(self):
        ''' Hermitian and anti-Hermitian part '''
        return (self.real, self.imag)

    def inner(self, other):
        ''' inner product between self and other (A · B)
            A · B = Tr (A^H @ B)
            assuming terms are orthonormal, i.e. 
            Tr(term_i^H @ term_j) = delta_{ij} 

            Input:
            other: Operator - the other operator to inner product with '''
        if len(other.terms) <= len(self.terms):
            shorter_terms, longer_terms = other.terms, self.terms
        else:
            shorter_terms, longer_terms = self.terms, other.terms
        result = 0
        for term in shorter_terms: # iterate through the shorter
            if term in longer_terms: # lookup in the longer
                result += self.terms[term].conjugate() * other.terms[term]
        return result

    def trace(self):
        ''' operator trace (Tr A)
            returns the coefficient in front of the identity
            operator component in A 
            (note that the trace here is normalized, i.e. 
             it equals Tr A / Tr I ) '''
        if () in self.terms:
            return self.terms[()]
        else:
            return 0

    def norm(self):
        ''' operator norm (sqrt(A · A) = sqrt(Tr (A^H @ A))) '''
        return math.sqrt(sum(abs(coef)**2 for coef in self.terms.values()))
    
    def normalize(self):
        ''' returns the normalized operator 
            A -> A / norm(A) '''
        norm = self.norm()
        if norm > self.tol:
            return self / norm
        else:
            return zero(self)

    def weight(self, weight_func=len):
        ''' operator weight (expected size of operator) 
            treat abs(coef)**2 as a density measure, evaluate
            the weighted average of terms under the mapping of
            a weighting function (which maps each term to a real
            number, such as the length of the term). Useful
            for determining the size of operator. '''
        numer = 0
        denom = 0
        for term in self.terms:
            p = abs(self.terms[term])**2
            numer += weight_func(term) * p
            denom += p
        if denom > 0:
            return numer / denom
        else: # if operator is empty, return 0
            return 0 

    def __round__(self, n=None):
        ''' rounding coefficients (round(A)) 
            Input:
            n: int (optional) - round to the nth decimals '''
        def gaussian_round(c):
            if isinstance(c, complex):
                return complex(round(c.real, n), round(c.imag, n))
            else:
                return round(c, n)
        return type(self)({term: gaussian_round(coef) for term, coef in self.terms.items()})

# constructors of universal operators
def zero(optype=None):
    ''' construct the zero element of the associative algebra 
        (i.e. the identity of addition)
        Input:
        optype: type - the operator type (the operator algebra)
                Operator - operator type will be inferred from 
                           the operator instance. '''
    if optype is None:
        optype = Operator
    elif isinstance(optype, Operator):
        optype = type(optype)
    return optype({})

def unit(optype=None):
    ''' construct the unit element of the associative algebra 
        (i.e. the identity of multiplication)
        Input:
        optype: type - the operator type (the operator algebra)
                Operator - operator type will be inferred from 
                           the operator instance. '''
    if optype is None:
        optype = Operator
    elif isinstance(optype, Operator):
        optype = type(optype)
    return optype({(): 1})


class MajoranaOperator(Operator):
    ''' Majorana operator in Clifford algebra 

        Parameters:
        terms: dict - {term: coef, ...} dictionary 
            term: product of Clifford generator (labeled by indices)
                  e.g. a term (0,4,5) dentotes χ0 χ4 χ5 

        Attributes:
        _parity: cache for fermion parity
        _loc_terms: cache for local term map '''
    def __init__(self, terms=None):
        super().__init__(terms)
        self._fermion_parity = None
        self._loc_terms = None

    def term_repr(self, term):
        ''' redefine representation of a term '''
        if len(term) == 0:
            return 'I '
        txt = ''
        for i in term:
            txt += 'χ{:d} '.format(i)
        return txt

    def term_mul(self, term1, term2):
        ''' redefine term-level multiplication rule '''
        if len(term1) == 0:
            return type(self)({term2: 1})
        if len(term2) == 0:
            return type(self)({term1: 1})
        n1 = len(term1) # length of term1
        n2 = len(term2) # length of term2
        i1 = 0 # term1 pointer
        i2 = 0 # term2 pointer
        ex = 0 # number of exchanges
        term = [] # to collect indices in the resulting term
        while i1 < n1 and i2 < n2:
            ind1 = term1[i1]
            ind2 = term2[i2]
            if ind1 == ind2: # indices collide
                ex += n1 - i1 - 1
                i1 += 1
                i2 += 1
            else:
                if ind1 < ind2:
                    term.append(ind1)
                    i1 += 1
                else: # ind1 > ind2
                    ex += n1 - i1
                    term.append(ind2)
                    i2 += 1
        if i1 < n1: # if term1 not exhausted
            term += term1[i1:] # dump the rest
        if i2 < n2: # if term2 not exhausted
            term += term2[i2:] # dump the rest
        term = tuple(term) # convert list to tuple
        sign = 1 - 2 * (ex % 2) # exchange sign
        return type(self)({term: sign})

    def term_adj_sign(self, term):
        ''' redefine term adjoint sign 
            product of Clifford generators are either Hermitian
            or anti-Hermitian, depending of the number of Clifford
            generators in the product. '''
        if (len(term) // 2) % 2 == 0:
            return 1
        else:
            return -1

    @property
    def parity(self):
        ''' fermion parity (Z2 grading of Clifford algebra) 
            +1: even parity (even grading)
            -1: odd parity (odd grading)
             0: mixed parity (no specific grading) '''
        if self._parity is None:
            # calculate fermion parity by inspecting each term
            for term in self.terms:
                term_parity = 1 - 2 * (len(term) % 2)
                if self._parity is None:
                    self._parity = term_parity
                else:
                    if self._parity != term_parity:
                        self._parity = 0
                        break
            if self._parity is None: # terms is empty, zero operator
                self._parity = 1 # treated as even parity (0*I)
        return self._parity

    @property
    def loc_terms(self):
        ''' map from local site to the set of covering terms 
            it has the structure of a dict whose values are sets
            {ind: {term, ...}, ...} 
            Knowing the locality structure helps to speed up 
            the calculation of commutator. '''
        if self._loc_terms is None:
            self._loc_terms = {}
            for term in self.terms:
                for i in term:
                    if i in self._loc_terms:
                        self._loc_terms[i].add(term)
                    else:
                        self._loc_terms[i] = {term}
        return self._loc_terms

    def commutate(self, other):
        ''' commutator of self with other 
            [A, B] = A @ B - B @ A
            Input:
            other: Operator - the operator to commutate with '''
        # commutator is localizable if either operator is even parity
        if self.parity == 1 or other.parity == 1:
            if len(other.terms) <= len(self.terms):
                shorter, longer = other, self
                sign = -1
            else:
                shorter, longer = self, other
                sign = 1
            result = zero(self)
            # single loop through shorter terms
            for term1 in shorter.terms:
                terms = set()
                for i in term1:
                    terms |= longer.loc_terms.get(i, set())
                for term2 in terms:
                    coef = shorter.terms[term1] * longer.terms[term2] * sign
                    term_comm = self.term_comm(term1, term2)
                    term_comm *= coef
                    result += term_comm
            return result
        else: # fall back to double loop
            result = zero(self)
            for term_self in self.terms:
                for term_other in other.terms:
                    coef = self.terms[term_self] * other.terms[term_other]
                    term_comm = self.term_comm(term_self, term_other)
                    term_comm *= coef
                    result += term_comm
            return result

    def term_comm(self, term1, term2):
        ''' redefine term-level commutation rule '''
        if len(term1) == 0:
            return zero(self)
        if len(term2) == 0:
            return zero(self)
        n1 = len(term1) # length of term1
        n2 = len(term2) # length of term2
        i1 = 0 # term1 pointer
        i2 = 0 # term2 pointer
        ex1 = 0 # number of exchanges in term1 @ term2
        ex2 = 0 # number of exchanges in term2 @ term1
        term = [] # to collect indices in the resulting term
        while i1 < n1 and i2 < n2:
            ind1 = term1[i1]
            ind2 = term2[i2]
            if ind1 == ind2: # indices collide
                ex1 += n1 - i1 - 1
                ex2 += n2 - i2 - 1
                i1 += 1
                i2 += 1
            else:
                if ind1 < ind2:
                    ex2 += n2 - i2
                    term.append(ind1)
                    i1 += 1
                else:
                    ex1 += n1 - i1
                    term.append(ind2)
                    i2 += 1
        if i1 < n1: # if term1 not exhausted
            term += term1[i1:] # dump the rest
        if i2 < n2: # if term2 not exhausted
            term += term2[i2:] # dump the rest
        term = tuple(term) # convert list to tuple
        # exchange signs
        sign1 = 1 - 2 * (ex1 % 2)
        sign2 = 1 - 2 * (ex2 % 2)
        sign = sign1 - sign2
        if sign == 0:
            return zero(self)
        else:
            return type(self)({term: sign})

def maj(*args):
    ''' Majorana operator constructor '''
    if len(args) != 1:
        return maj(args)
    else:
        term = args[0]
        if isinstance(term, int):
            return maj((term,))
        if isinstance(term, tuple):
            return MajoranaOperator({term: 1})
        elif isinstance(term, list):
            return maj(tuple(term))
        else:
            raise NotImplementedError("majorana constructor is not implemented for '{}'".format(type(term).__name__))


 