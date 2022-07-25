import math
import numpy
import cvxopt
from numba import njit
from numba.typed import List

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
    tol = 1.e-10
    _H = None    # cache Hermitian conjugate
    _real = None # cache Hermitian part
    _imag = None # cache anti-Hermitian part

    def __init__(self, terms=None):
        self.terms = {} if terms is None else terms

    # ---- representation ----
    def __repr__(self, max_terms=32):
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
        term_count = 0
        for term in self.terms:
            txt_term = _coef_repr(self.terms[term])
            txt_term += self.term_repr(term)
            if txt != '' and txt_term[0] != '-':
                txt += '+'
            txt += txt_term
            term_count += 1
            if term_count > max_terms:
                txt += '...'
                break
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

    def __len__(self):
        ''' number of terms in the operator '''
        return len(self.terms)

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
        elif isinstance(other, Operator):
            return self @ other # promote to operator multiplication
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
            other = identity(self) * other # non-operators are treated as numbers
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
            other = identity(self) * other # non-operators are treated as numbers
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
            other = identity(self) * other # non-operators are treated as numbers
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
    # !!! to be redefined by specific Operator subclasses
    def __matmul__(self, other):
        ''' operator multiplication (A @ B)
            Input:
            other: Operator - the operator to mutiply '''
        result = zero(self)
        for term_self in self.terms:
            for term_other in other.terms:
                term, coef = term_self + term_other, 1
                coef *= self.terms[term_self] * other.terms[term_other]
                result += type(self)({tuple(term):coef})
        return result

    def commutate(self, other):
        ''' commutator of self with other 
            [A, B] = A @ B - B @ A
            Input:
            other: Operator - the operator to commutate with '''
        return self @ other - other @ self

    # ----  inner product structure ----
    def conjugate(self):
        ''' Hermitian conjugate of an operator '''
        return type(self)({term: self.term_conj_sign(term) * coef.conjugate() for term, coef in self.terms.items()})

    # !!! to be redefined by specific Operator subclasses
    def term_conj_sign(self, term):
        ''' term conjugate sign 
            the additional sign generated when taking the conjugate
            of a term (this happens when the term is anti-Hermitian)
            Input:
            term: tuple - the term in consideration '''
        return 1

    @property
    def H(self):
        ''' Hermitian conjugation '''
        if self._H is None:
            self._H = self.conjugate()
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
        return Operators([self.real, self.imag])

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

    def krylov(self, n=None, show_progress=False):
        ''' construct the Krylov space generated by the operator 
            Input:
            n: int (optional) - upto the nth power of the operator. '''
        basis = [identity(self)] # collect basis operator (starting from identity)
        alphas = [] # collect diagonal elements of Lanczos matrix
        betas = []  # collect off-diagonal elements of Lanczos matrix
        if show_progress: 
            print('     len  weight')
            print(f'{len(basis)-1:2d}: {1:4d} {0.:6.1f}')
        if n is None:
            n = numpy.inf
        while n >= 0:
            new = self @ basis[-1]
            alpha = basis[-1].inner(new)
            new = new - alpha * basis[-1]
            alphas.append(alpha)
            if len(basis) > 1: # if basis[-2] can be accessed
                beta  = basis[-2].inner(new)
                new = new - beta * basis[-2]
                betas.append(beta)
            if n > 0: # if iteration is not ending
                norm = new.norm()
                if norm > self.tol:
                    new /= norm
                    basis.append(new)
                    if show_progress: print(f'{len(basis)-1:2d}: {len(new):4d} {new.weight():6.1f}')
                else: # if Krylov space exhausted
                    break
            n -= 1 # max power decrease
        # construct Lanczos matrix
        mat = numpy.diag(alphas, 0) + numpy.diag(betas, 1) + numpy.diag(betas, -1).conjugate()
        n = len(alphas) # actual dimension of Krylov (sub)space
        ope = [] # collect OPE tensor
        power_mat = numpy.eye(n, dtype=mat.dtype)
        vec = numpy.zeros(n, dtype=mat.dtype)
        vec[0] = 1
        for k in range(n):
            ope.append(power_mat)
            for j in range(k):
                ope[k] = ope[k] - vec[j] * ope[j]
            ope[k] /= vec[k]
            power_mat = numpy.matmul(mat, power_mat)
            vec = numpy.matmul(mat, vec)
        ops = OperatorSpace(basis)
        ops._ope = numpy.stack(ope, axis=1)
        return ops



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

def identity(optype=None):
    ''' construct the identity element of the associative algebra 
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
    '''
    _parity = None    # cache fermion parity
    _loc_terms = None # cache local term map

    def term_repr(self, term):
        ''' redefine representation of a term '''
        if len(term) == 0:
            return 'I '
        txt = ''
        for i in term:
            txt += 'χ{:d} '.format(i)
        return txt

    def __matmul__(self, other):
        ''' operator multiplication (A @ B)
            Input:
            other: Operator - the operator to mutiply '''
        result = zero(self)
        for term_self in self.terms:
            for term_other in other.terms:
                term, coef = maj_term_mul(term_self, term_other)
                coef *= self.terms[term_self] * other.terms[term_other]
                result += type(self)({tuple(term):coef})
        return result

    def term_conj_sign(self, term):
        ''' redefine term conjugate sign 
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
                    term, coef = maj_term_comm(term1, term2)
                    coef *= shorter.terms[term1] * longer.terms[term2] * sign
                    result += type(self)({tuple(term):coef})
            return result
        else: # fall back to double loop
            result = zero(self)
            for term_self in self.terms:
                for term_other in other.terms:
                    term, coef = maj_term_comm(term_self, term_other)
                    coef *= self.terms[term_self] * other.terms[term_other]
                    result += type(self)({tuple(term):coef})
            return result

@njit
def maj_term_mul(term1, term2):
    ''' term-level multiplication for MajoranaOperator '''
    if len(term1) == 0: # term1 is identity operator
        return term2, 1
    if len(term2) == 0: # term2 is identity operator
        return term1, 1
    n1 = len(term1) # length of term1
    n2 = len(term2) # length of term2
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    ex = 0 # number of exchanges
    term = [0][:0] # empty term (this allows numba type inference)
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
    for i in range(i1, n1): # if term1 not exhausted
        term.append(term1[i]) # dump the rest
    for i in range(i2, n2): # if term2 not exhausted
        term.append(term2[i]) # dump the rest
    sign = 1 - 2 * (ex % 2) # exchange sign
    return term, sign

@njit
def maj_term_comm(term1, term2):
    ''' term-level commutation for MajoranaOperator '''
    if len(term1) == 0: # term1 is identity operator
        return [0][:0], 0 
    if len(term2) == 0: # term2 is identity operator
        return [0][:0], 0
    n1 = len(term1) # length of term1
    n2 = len(term2) # length of term2
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    ex1 = 0 # number of exchanges in term1 @ term2
    ex2 = 0 # number of exchanges in term2 @ term1
    term = [0][:0] # empty term (this allows numba type inference)
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
    for i in range(i1, n1): # if term1 not exhausted
        term.append(term1[i]) # dump the rest
    for i in range(i2, n2): # if term2 not exhausted
        term.append(term2[i]) # dump the rest
    # exchange signs
    sign1 = 1 - 2 * (ex1 % 2)
    sign2 = 1 - 2 * (ex2 % 2)
    sign = sign1 - sign2
    if sign == 0:
        return [0][:0], 0
    else:
        return term, sign

def maj(*args):
    ''' Majorana operator constructor 
        Examples:
        >>> maj()
        I

        >>> maj(0)
        χ0

        >>> maj(1,2)
        χ1 χ2 
        
        >>> maj([2,3])
        χ2 χ3
    '''
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

class PauliOperator(Operator):
    ''' Pauli operator in Pauli algebra 

        Parameters:
        terms: dict - {term: coef, ...} dictionary 
            term: product of Pauli operators 
                  (labeled by (index, operator) pairs)
                  e.g. a term ((0,1), (2,3)) dentotes X0 Z2 
    '''
    _loc_terms = None # cache local term map

    def term_repr(self, term):
        ''' redefine representation of a term '''
        opnames = ('I','X','Y','Z')
        if len(term) == 0:
            return 'I'
        txt = ''
        for i, a in term:
            txt += opnames[a] + '{:d} '.format(i)
        return txt

    def __matmul__(self, other):
        ''' operator multiplication (A @ B)
            Input:
            other: Operator - the operator to mutiply '''
        result = zero(self)
        for term_self in self.terms:
            for term_other in other.terms:
                term, coef = pauli_term_mul(term_self, term_other)
                coef *= self.terms[term_self] * other.terms[term_other]
                result += type(self)({tuple(term):coef})
        return result

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
                for i, a in term:
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
            for i, _ in term1:
                terms |= longer.loc_terms.get(i, set())
            for term2 in terms:
                term, coef = pauli_term_mul(term1, term2)
                if coef.imag != 0:
                    coef *= 2 * shorter.terms[term1] * longer.terms[term2] * sign
                    result += type(self)({tuple(term):coef})
        return result

@njit
def pauli_term_mul(term1, term2):
    ''' term-level multiplication for PauliOperator '''
    if len(term1) == 0: # term1 is identity operator
        return term2, 1
    if len(term2) == 0: # term2 is identity operator
        return term1, 1
    n1 = len(term1) # length of term1
    n2 = len(term2) # length of term2
    i1 = 0 # term1 pointer
    i2 = 0 # term2 pointer
    term = [(0,0)][:0] # empty term (this allows numba type inference)
    phase = 1 # to track the phase factor
    pauli_rule = [0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0]
    phase_rule = [1+0j,1,1,1,1,1,1j,-1j,1,-1j,1,1j,1,1j,-1j,1]
    while i1 < n1 and i2 < n2:
        ind1, mu1 = term1[i1]
        ind2, mu2 = term2[i2]
        if ind1 == ind2: # indices collide
            mu12 = 4 * mu1 + mu2
            mu = pauli_rule[mu12]
            # if mu == 0: identity operator ignored
            if mu != 0:
                phase *= phase_rule[mu12]
                term.append((ind1, mu))
            i1 += 1
            i2 += 1
        else:
            if ind1 < ind2:
                term.append((ind1, mu1))
                i1 += 1
            else: # ind1 > ind2
                term.append((ind2, mu2))
                i2 += 1
    for i in range(i1, n1): # if term1 not exhausted
        term.append(term1[i]) # dump the rest
    for i in range(i2, n2): # if term2 not exhausted
        term.append(term2[i]) # dump the rest
    return term, phase

def pauli(*args):
    ''' Pauli operator constructor 
        Examples:
        >>> pauli()
        I

        >>> pauli(0), pauli(1), pauli(2), pauli(3)
        (I, X0, Y0, Z0)

        >>> pauli('I'), pauli('X'), pauli('Y'), pauli('Z') 
        (I, X0, Y0, Z0)
        
        >>> pauli('-X'), pauli('iX'), pauli('-iX')
        (- X0, i X0, -i X0)

        >>> pauli('X3 Z5')
        X3 Z5

        >>> pauli('IIIXIZ')
        X3 Z5

        >>> pauli('I2XIZ')
        X3 Z5

        >>> pauli([0,0,0,1,0,3])
        X3 Z5

        >>> pauli({3:'X', 5:'Z'})
        X3 Z5

        >>> pauli({3:1, 5:3})
        X3 Z5

        >>> pauli(((3,1), (5,3)))
        X3 Z5
        
        >>> pauli(((3,'X'), (5,'Z')))
        X3 Z5

        >>> pauli((('X',3), ('Z',5)))
        X3 Z5

        >>> pauli('X',3,'Z',5)
        X3 Z5
        '''
    # reduce arguments
    if len(args) != 1:
        return pauli(args)
    else:
        obj = args[0]
        if isinstance(obj, int):
            return pauli((obj,))
        elif isinstance(obj, str):
            return pauli(list(obj))
        elif isinstance(obj, dict):
            return pauli(tuple(obj.items()))
        elif isinstance(obj, (tuple, list)):
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
                    if isinstance(item[1], int):
                        # item is of the form ('Z',1)
                        i = item[1] # record position
                elif isinstance(item[0], int):
                    i, a = item
                    if isinstance(a, str):
                        # item is of the form (1,'Z')
                        a = opname.get(a, None) # try get operator name
                    elif isinstance(a, int):
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
        elif isinstance(item, int):
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


class Operators(numpy.ndarray):
    ''' Represent an array of operators (subclass form numpy.ndarray)
        
        Parameter:
        ops: list, numpy.ndarray -  an array of operators 
             (each element is an Operator object)
             (to be parsed by numpy array constructor) '''
    def __new__(cls, ops):
        return numpy.asarray(ops).view(cls).squeeze()

    def __repr__(self, max_line_width=75):
        prefix = type(self).__name__ + '('
        suffix = ')'
        if self.size > 0 or self.shape == (0,):
            lst = numpy.array2string(self, max_line_width=max_line_width, 
                separator=', ', prefix=prefix, suffix=suffix)
        else: # show zero-length shape unless it is (0,)
            lst = "[], shape=%s" % (repr(arr.shape),)
        return prefix + lst + suffix

    def squeeze(self):
        ''' fall back to Operator if dimensionless '''
        if self.ndim == 0: # if Operators becomes dimensionless
            return self.item() # return item
        return self 

    def append(self, other):
        return type(self)(numpy.append(self, other))

    def asarray(self):
        ''' convert object array as numerical array
            caller must ensure array elements are numerical '''
        try: # try convert to float
            return numpy.asfarray(self)
        except TypeError: # if not, try complex
            return numpy.asarray(self, dtype='complex')

    def __matmul__(self, other):
        return super(Operators, self).__matmul__(other).squeeze()

    def adjoint(self, other):
        ''' adjoint action by a Lie algebra generator g 
            
            Input:
            other: Operator - Lie algebra generator g
            
            Output:
            ops: Operators - i[g, x] for every operator x in 
                             the operator array '''
        return numpy.frompyfunc(lambda x: 1j*other.commutate(x),1,1)(self)

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

    def inner(self, other):
        return numpy.frompyfunc(lambda x,y: x.inner(y),2,1)(self, other).asarray()

    def outer(self, other):
        return numpy.frompyfunc(lambda x,y: x@y,2,1).outer(self, other).view(Operators)

    def trace(self):
        return numpy.frompyfunc(lambda x: x.trace(),1,1)(self).asarray()

    def norm(self):
        return numpy.frompyfunc(lambda x: x.norm(),1,1)(self).asarray()

    def normalize(self, tol=1.e-10):
        return numpy.frompyfunc(lambda x: x.normalize(tol),1,1)(self)

    def weight(self, weight_func=len):
        return numpy.frompyfunc(lambda x: x.weight(weight_func),1,1)(self).asarray()

    def round(self, n=None):
        return numpy.frompyfunc(lambda x: round(x, n),1,1)(self)

class OperatorSpace(Operators):
    ''' Represent a operator space spanned by a set of operator basis
        (subclass of Operators, listing the basis operators)
        (basis operators are assumed to be orthonormal) 

        Parameter:
        basis: list, numpy.ndarray - a list of basis operators.
                (each element is a Operator object) 
    '''
    _ope = None # cache OPE tensor

    @property
    def ope(self):
        if self._ope is None:
            self._ope = self.represent(self.outer(self), axis=0)
        return self._ope
    
    def represent(self, other, axis=-1):
        ''' represent operator or operator array in the operator space 
            Input:
            other: Operator, Operators - operator(s) to represent 
            axis: int - the axis in which the vector representation resides

            Output: numpy.ndarray - vector representation(s) '''
        if isinstance(other, Operator):
            return self.inner(other) # ignore axis
        # assuming other is Operators instance
        axis = axis % (other.ndim + 1)
        other_shape = list(other.shape)
        other_shape.insert(axis, 1)
        basis_shape = [1]*other.ndim
        basis_shape.insert(axis,-1)
        result = self.reshape(basis_shape).inner(other.reshape(other_shape))
        return result

    # reconstruct operator from its representation
    # axis - specify the axis in which the vector representation resides
    def reconstruct(self, other, axis=-1):
        ''' reconstruct operator from its representation 
            Input:
            other: numpy.ndarray - vector representation(s)
            axis: int - the axis in which the vector representation resides

            Output: Operator, Operators - reconstructed operator(s) '''
        if other.ndim == 1:
            return self.dot(other) # ignore axis
        return numpy.tensordot(self, other, axes=([0],[axis])).view(Operators)

    
def qboot(H, n=None, show_progress=False):
    ''' quantum bootstrap 
        Input:
        H: Operator - Hamiltonian operator 

        Output:
        rho: Operator - ground state density matrix '''
    space = H.krylov(n, show_progress)
    if show_progress: print('Krylov space constructed.')
    obj_vector = space.represent(H).real[1:,None]
    sdp_matrix = space.ope.real[0]
    sdp_tensor = -space.ope.real[1:]
    sdp_tensor = sdp_tensor.reshape(sdp_tensor.shape[0], -1).T
    cvxopt.solvers.options['show_progress'] = show_progress
    sol = cvxopt.solvers.sdp(cvxopt.matrix(obj_vector), 
                         hs=[cvxopt.matrix(sdp_matrix)],
                         Gs=[cvxopt.matrix(sdp_tensor)])
    sol_vector = numpy.array(sol['x']).flatten()
    sol_vector = numpy.concatenate([numpy.array([1]), sol_vector])
    rho = space.reconstruct(sol_vector)
    return rho



 


