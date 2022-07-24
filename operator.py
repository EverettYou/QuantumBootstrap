import torch

class Operator():
    ''' Represent a quantum operator as a liner superposition of 
        terms. Eeach term (monomial operator) is specified by an
        array of tokens.
        
        Parameters:
        terms: torch.Tensor - a list of operator terms
        coefs: torch.Tensor - a list of coefficients (generally complex) '''
    def __init__(self, terms, coefs):
        self.terms = terms # (len, N)
        self.coefs = coefs # (len, )

    # ---- representation ----
    def __repr__(self, max_terms=16):
        ''' operator representation '''
        with torch.no_grad():
            txt = ''
            dots = ''
            coefs = self.coefs
            if coefs.shape[0] > max_terms:
                coefs = coefs[:max_terms]
                dots = '...'
            for k, c in enumerate(coefs):
                if c != 0:
                    txt_term = self._c_repr(c) + self._term_repr(self.terms[k])
                    if txt != '' and txt_term[0] != '-':
                        txt += '+'
                    txt += txt_term
            if txt == '':
                txt = '0'
            else:
                txt += dots
            txt = txt.strip()
        if self.coefs.grad_fn is not None:
            txt += ' (grad_fn={})'.format(type(self.coefs.grad_fn))
        elif self.coefs.requires_grad:
            txt += ' (requires_grad=True)'
        return txt

    def _c_repr(self, c):
        ''' constant number representation 
            Input: c: complex - a number 
            Output: txt: str - string representation '''
        if c.imag == 0.:
            c = c.real
            if c == torch.floor(c):
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
            if c == torch.floor(c):
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

    def _token_repr(self, tok):
        ''' token representation 
            Input: tok: int - an integer token
            Output: txt: str - string representation of token '''
        txt = f'<{tok:d}>'
        return txt

    def _term_repr(self, term):
        ''' tokens representation 
            Input: term: torch.Tensor (N,) - a sequence of tokens 
            Output: txt: str - string representation of tokens '''
        txt = ''
        for i, tok in enumerate(term):
            if tok != 0:
                txt += self._token_repr(tok) + f'{i:d} '
        if txt == '':
            txt = 'I '
        return txt

    # ---- terms shape ----
    def __len__(self):
        ''' number of terms in the operator '''
        return self.terms.shape[0]

    @property
    def N(self):
        ''' number of sites/qubits '''
        return self.terms.shape[1]

    def _pad(self, other):
        if self.N < other.N:
            self.terms = torch.nn.functional.pad(self.terms, (0,other.N-self.N))
        elif self.N > other.N:
            other.terms = torch.nn.functional.pad(other.terms, (0,self.N-other.N))

    # ---- autograd ----
    @property
    def requires_grad(self):
        return self.coefs.requires_grad
    
    def requires_grad_(self, requires_grad=True):
        self.coefs.requires_grad_(requires_grad=requires_grad)
        return self
    
    @property
    def grad(self):
        ''' represent coefficient gradient as a gradient operator '''
        grad = self.coefs.grad
        if grad is None:
            return None
        else:
            return Operator(self.terms, grad)
    
    # ---- device ----
    def to(self, device):
        self.terms = self.terms.to(device)
        self.coefs = self.coefs.to(device)
        return self

    # ---- algebra ----
    def __mul__(self, other):
        ''' scalar multiplication '''
        if other == 0:
            return type(self)(self.terms[[]], self.coefs[[]])
        else:
            return type(self)(self.terms, other*self.coefs)

    def __rmul__(self, other):
        ''' scalar multiplication (right) '''
        return self * other

    def __truediv__(self, other):
        ''' scalar division '''
        return self * (1/other)

    def __neg__(self):
        return type(self)(self.terms, -self.coefs)

    def __add__(self, other):
        if isinstance(other, Operator):
            self._pad(other)  
            terms = torch.cat([self.terms, other.terms])
            coefs = torch.cat([self.coefs, other.coefs])
            return type(self)(terms, coefs).reduce()
        else:
            result = self + other * identity(self)
            return result.reduce()

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)

    def __eq__(self, other):
        if other == 0:
            return (self.coefs == 0).all()
        else:
            return self - other == 0

    def reduce(self, tol=1e-10):
        ''' Reduce the operator by:
            1. combine similar terms
            2. drop terms that are too small (abs < tol) '''
        terms, inv_indx = torch.unique(self.terms, dim=0, return_inverse=True)
        coefs = torch.zeros((terms.shape[0],)).to(self.coefs)
        coefs.scatter_add_(0, inv_indx, self.coefs)
        mask = coefs.abs() > tol
        return type(self)(terms[mask], coefs[mask])

    @property
    def trace(self):
        ''' compute Tr O '''
        mask = torch.all(self.terms == 0, -1)
        return self.coefs[mask].sum()

    def traceless(self):
        return self - self.trace * identity(self)

    @property
    def H(self):
        # Hermitian conjugation of this operator O -> O^†
        return type(self)(self.terms, self.coefs.conj())

def identity(op):
    ''' construct identity operator 
        Input: op: Operator - identity operator of the same shape '''
    terms = torch.zeros((1, op.N)).to(op.terms)
    coefs = torch.ones((1,)).to(op.coefs)
    return type(op)(terms, coefs)

class PauliOperator(Operator):
    ''' Represent a Pauli operator '''
    token_rule = torch.tensor(
        [[0,1,2,3],
         [1,0,3,2],
         [2,3,0,1],
         [3,2,1,0]]).flatten()
    coeff_rule = torch.tensor(
        [[1,  1,  1,  1],
         [1,  1, 1j,-1j],
         [1,-1j,  1, 1j],
         [1, 1j,-1j,  1]]).flatten()
    token_names = ['I', 'X', 'Y', 'Z']
    def __init__(self, terms, coefs):
        self.terms = terms
        self.coefs = coefs

    def _token_repr(self, tok):
        ''' token representation 
            Input: tok: int - an integer token
            Output: txt: str - string representation of token '''
        txt = self.token_names[tok]
        return txt

    def __matmul__(self, other):
        ''' define: A @ B = A B '''
        self._pad(other) 
        token_prod = 4 * self.terms.unsqueeze(1) + other.terms.unsqueeze(0) # (n1, n2, N)
        terms = self.token_rule.to(token_prod.device)[token_prod].view(-1, self.N) # (n1*n2, N)
        coefs = self.coefs.unsqueeze(1) * other.coefs.unsqueeze(0) # (n1, n2)
        coefs = coefs.view(-1) # (n1*n2,)
        coefs *= self.coeff_rule.to(token_prod.device)[token_prod].prod(-1).view(-1) # (n1*n2,)
        return type(self)(terms, coefs).reduce()

def pauli(obj, N=None):
    if isinstance(obj, torch.Tensor):
        terms = obj.view(1,-1)
    else:
        if isinstance(obj, (tuple, list)):
            N = len(obj)
            inds = enumerate(obj)
        elif isinstance(obj, dict):
            if N is None:
                raise ValueError('pauli(inds, N) must specify qubit number N when inds is dict.')
            inds = obj.items()
        elif isinstance(obj, str):
            return pauli(list(obj))
        else:
            raise TypeError('pauli(obj) recieves obj of type {}, which is not implemented.'.format(type(obj).__name__))
        terms = torch.zeros(1, N, dtype=torch.long)
        for i, p in inds:
            assert i < N, 'Index {} out of bound {}'.format(i, N)
            if p == 'I':
                p = 0
            elif p == 'X':
                p = 1
            elif p == 'Y':
                p = 2
            elif p == 'Z':
                p = 3
            terms[0, i] = p 
    coefs = torch.ones(1, dtype=torch.cfloat)
    return PauliOperator(terms, coefs)

class MajoranaOperator(Operator):
    ''' Represent a Majorana operator '''
    token_names = ['I', 'χ']
    def __init__(self, terms, coefs):
        self.terms = terms
        self.coefs = coefs

    def _token_repr(self, tok):
        ''' token representation 
            Input: tok: int - an integer token
            Output: txt: str - string representation of token '''
        txt = self.token_names[tok]
        return txt

    def __matmul__(self, other):
        ''' define: A @ B = A B '''
        self._pad(other)
        terms = self.terms.unsqueeze(1) + other.terms.unsqueeze(0) # (n1, n2, N)
        terms = (terms%2).view(-1, self.N) # (n1*n2, N)
        coefs = self.coefs.unsqueeze(1) * other.coefs.unsqueeze(0) # (n1, n2)
        coefs = coefs.view(-1) # (n1*n2,)
        mark = torch.cat([self.terms[:,1:], torch.zeros_like(self.terms[:,:1])],dim=-1) # (n1, N)
        sign = mark.unsqueeze(1) * other.terms.cumsum(-1).unsqueeze(0) # (n1, n2, N)
        sign = sign.view(-1, self.N).sum(-1) # (n1*n2,)
        coefs *= (-1)**sign # (n1*n2,)
        return type(self)(terms, coefs).reduce()

    @property
    def H(self):
        sign = (self.terms.sum(-1)//2)%2
        return type(self)(self.terms, self.coefs.conj() * (-1)**sign)
    

def maj(*args, N=None):
    if len(args) != 1:
        return maj(args)
    else:
        arg = args[0]
        if isinstance(arg, int):
            return maj((arg,))
        elif isinstance(arg, tuple):
            N = max(arg)+1 if N is None else N
            terms = torch.zeros(1, N, dtype=torch.long)
            for i in arg:
                terms[0, i] = 1
            coefs = torch.ones(1, dtype=torch.cfloat)
            return MajoranaOperator(terms, coefs)

