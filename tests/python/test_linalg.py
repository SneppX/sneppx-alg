import numpy as np

from SneppX_ALG import linalg as la


def test_lu():
    A = np.array([[4.0, 3.0], [6.0, 3.0]])
    P, L, U = la.lu(A)
    assert np.allclose(P @ L @ U, A)
    assert np.allclose(L, np.tril(L))


def test_qr():
    A = np.random.default_rng(0).random((4, 3))
    Q, R = la.qr(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-8)


def test_svd():
    A = np.array([[3.0, 0.0], [0.0, -2.0]])
    U, s, Vt = la.svd(A)
    assert np.allclose(np.sort(s)[::-1], [3.0, 2.0])
    assert np.allclose(U @ np.diag(s) @ Vt, A)


def test_cholesky_eig():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    L = la.cholesky(A)
    assert np.allclose(L @ L.T, A)
    w, V = la.eigh(A)
    assert np.allclose(A @ V, V * w)


def test_solve_and_inverse():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = la.solve(A, b)
    assert np.allclose(A @ x, b)
    assert np.allclose(la.inv(A) @ A, np.eye(2))
    assert np.allclose(la.inv(A), la.pinv(A))


def test_lstsq():
    A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0, 3.0])
    x, *_ = la.lstsq(A, b)
    assert np.allclose(A @ x, b, atol=1e-6)


def test_solve_triangular():
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    b = np.array([2.0, 4.0, 15.0])
    x = la.solve_triangular(A, b, lower=True)
    assert np.allclose(A @ x, b)
    x2 = la.solve_triangular(
        np.array([[2.0, 1.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]),
        b,
        lower=False,
    )
    assert np.allclose(
        np.array([[2.0, 1.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]]) @ x2, b
    )


def test_norms():
    x = np.array([3.0, 4.0])
    assert abs(la.vector_norm(x) - 5.0) < 1e-9
    A = np.diag([3.0, 1.0])
    assert abs(la.frobenius_norm(A) - np.sqrt(10.0)) < 1e-9
    assert abs(la.spectral_norm(A) - 3.0) < 1e-9
    assert abs(la.nuclear_norm(A) - 4.0) < 1e-9


def test_utils():
    A = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]])
    assert np.allclose(la.tril(A), [[1, 0, 0], [0, 4, 0]])
    assert np.allclose(la.triu(A), [[1, 2, 3], [0, 4, 5]])
    assert abs(la.det(np.eye(3)) - 1.0) < 1e-12
    sign, logdet = la.slogdet(2 * np.eye(2))
    assert sign == 1.0 and abs(logdet - np.log(4.0)) < 1e-9
    assert la.matrix_rank(A) == 2


def test_symmetry_predicates():
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert la.is_symmetric(S)
    assert la.is_positive_definite(S)
    assert not la.is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
