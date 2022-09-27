import numpy as np
from scipy.stats import norm
N = norm.cdf


# noinspection PyUnresolvedReferences,SpellCheckingInspection,PyPep8Naming
def basket_approx(opt_type: str, T: float, r: float, strike: float,
                  fwds, sigmas, correls, weights=None,
                  dtype=np.float64):
    """
    Basket (and spread) general approximation
    assuming a lognormal distribution for each underlying forward:

    call_payoff = max(Sum_i^L w_i fwd_i - Sum_j^S w_j fwd_j - strike, 0.)
    where there are L long fwds and N short fwds
    put_payoff = max(strike - (Sum_i^L w_i fwd_i - Sum_j^S w_j fwd_j), 0.) by put-call parity

    i.e. the formula incorporates spread options, e.g. crack clean spread options by setting L=1 and S=2.

    :param opt_type: 'call' or 'put'
    :param T: time to expiry in years (e.g. 0.5 for 6 months)
    :param r: interest rate
    :param strike: strike of the option (>0)
    :param fwds: list of all current forward prices
    :param sigmas: list of volatilities for the forwards returns
    :param correls: correlation matrix
    :param weights: list of weights (positive for long assets, negative for short assets), default = 1.
    :param dtype:

    :return: (lower bound approx of) basket (and spread) prices (for lognormal underlying forwards)


    # SPREAD Example
    T = 1
    r = 0.05
    strike = 30 # ATM
    fwds = np.array([100, 24, 46])
    sigmas = [0.4, 0.22, 0.3]
    r12, r13, r23 = 0.91, 0.91, 0.43
    correls = [[1., r12, r13], [r12, 1, r23], [r13, r23, 1.]]
    weights = [1, -1., -1.]

    OR

    # BASKET Example
    T = 5 # 5 years
    r = 0.
    fwds = 100. * np.ones(4)
    sigmas = 0.4 * np.ones(4)
    correls = 0.5 * np.ones((4, 4))
    np.fill_diagonal(correls, 1.)
    weights = 0.25 * np.ones(4)
    strikes = 100

    call = basket_approx_py(opt_type='c', T=T, r=r, strike=k_i,
    fwds=fwds, sigmas=sigmas, correls=correls, weights=weights, dtype=np.float64)
    """

    # 1. transform and order inputs
    fwds = np.array(fwds, dtype=dtype)
    sigmas = np.array(sigmas, dtype=dtype)
    correls = np.array(correls, dtype=dtype)
    w = np.ones(fwds.shape[0], dtype=dtype) / fwds.shape[0] if weights is None else np.array(weights, dtype=dtype)

    diag_vol = np.diag(sigmas)
    correl_np = np.array(correls)
    cov_matrix = np.dot(np.dot(diag_vol, correl_np), diag_vol)

    # split arrays: L long from S short
    long = w > 0
    w_l, w_s = w[long], np.abs(w[~long])
    L, S = w_l.shape[0], w_s.shape[0]
    fwds_l, fwds_s = fwds[long], fwds[~long]

    cov_mat_ord1 = np.vstack((cov_matrix[long], cov_matrix[~long]))  # order rows
    cov_mat_ord = np.hstack((cov_mat_ord1[:, long], cov_mat_ord1[:, ~long]))  # order cols

    # pre-computed vars
    b_l, b_s = w_l * fwds_l, w_s * fwds_s
    e_F, e_K = np.sum(b_l), (np.sum(b_s) + strike)
    b_l, b_s = b_l / e_F, b_s / e_K

    df = np.exp(-r * T)  # by default dtype=np.float64
    T_sqrt = np.sqrt(T)

    # calculate volatility
    b_ord = np.hstack((b_l, -b_s))  # NB: minus
    var = np.dot(np.dot(b_ord, cov_mat_ord), b_ord)
    vol = np.sqrt(var)

    # calculate d_k
    long_var = np.dot(np.dot(b_l, cov_mat_ord[:L, :L]), b_l)
    short_var = np.dot(np.dot(b_s, cov_mat_ord[L:, L:]), b_s)
    d_k = (np.log(e_F) - np.log(e_K) - 0.5 * (long_var - short_var) * T) / (vol * T_sqrt)

    # calculate d_fwds
    d_fwds = (np.dot(cov_mat_ord[:, :L], b_l) - np.dot(cov_mat_ord[:, L:], b_s)) * T_sqrt / vol + d_k

    long_value = np.sum([w_l[k] * fwds_l[k] * N(d_fwds[k]) for k in range(L)])
    short_value = np.sum([w_s[k] * fwds_s[k] * N(d_fwds[L + k]) for k in range(S)])
    # intrinsic = np.sum(fwds * w) - strike * df

    call = df * (long_value - short_value - strike * N(d_k))
    call = np.maximum(0., call)  # np.maximum(np.maximum(intrinsic, 0.), call)

    if opt_type.lower() == 'c':
        return call
    elif opt_type.lower() == 'p':
        put = call - (np.sum(fwds * w) - strike) * df  # by put-call parity
        return np.maximum(0., put)  # np.maximum(np.maximum(-intrinsic, 0.), put)
    else:
        raise ValueError("select either 'call' or 'put' - tertium non datur.")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':

    def test_3spread():
        """
         test clean dark spread option price(log-normal approx)
            call payoff = max(F_1 - F_2 - F_3 - strike, 0.)
         """
        # params
        T = 1
        r = 0.05
        strike = 30  # ATM
        forwards = np.array([100, 24, 46])
        sigmas = [0.4, 0.22, 0.3]
        r12, r13, r23 = 0.9, 0.9, 0.9
        correls = [[1., r12, r13], [r12, 1, r23], [r13, r23, 1.]]
        weights = [1, -1., -1.]

        # 1. calculate prices
        call = basket_approx(opt_type='c', T=T, r=r, strike=strike,
                             fwds=forwards, sigmas=sigmas, correls=correls, weights=weights, dtype=np.float64)
        put = basket_approx(opt_type='p', T=T, r=r, strike=strike,
                            fwds=forwards, sigmas=sigmas, correls=correls, weights=weights)

        assert round(call, 4) == 9.0598  # expected call price 9.0598 - from formulae in Rikard Green 2015 paper
        assert round(put, 4) == 9.0598  # as above (sum fwd_i * w_i = strike -> call = put)

        # 2. Permute weights, sigmas and weights (unchanged correl mat) to check ordering in the function
        perm = [2, 0, 1]  # np.random.permutation(dim)
        sigmas, weights = np.array(sigmas), np.array(weights)
        forwards, sigmas, weights = forwards[perm], sigmas[perm], weights[perm]
        call_perm = basket_approx(opt_type='c', T=T, r=r, strike=strike,
                                  fwds=forwards, sigmas=sigmas, correls=correls, weights=weights, dtype=np.float64)
        put_perm = basket_approx(opt_type='p', T=T, r=r, strike=strike,
                                 fwds=forwards, sigmas=sigmas, correls=correls, weights=weights)

        assert round(call_perm, 4) == 9.0598
        assert round(put_perm, 4) == 9.0598


    # noinspection PyTypeChecker
    def test_basket():
        """
         test basket option price (log-normal approx)
            call payoff = max(Sum_i^L w_i fwd_i - strike, 0.)
            with weights_i = 0.25 for i=1, ..., 4
         """
        # params (in Krekel at al (2004), An Analysis of Pricing Methods for Basket Options)
        T = 5  # 5 years
        r = 0.
        fwds = 100. * np.ones(4)
        sigmas = 0.4 * np.ones(4)
        correls = 0.5 * np.ones((4, 4))
        np.fill_diagonal(correls, 1.)
        weights = 0.25 * np.ones(4)

        strikes = [80, 100, 120]
        expected_calls = [36.04, 27.63, 21.36]  # lower bounds from paper vs. simulated 36.35, 28.00, 21.76 from paper

        for i, k_i in enumerate(strikes):
            call = basket_approx(opt_type='c', T=T, r=r, strike=k_i,
                                 fwds=fwds, sigmas=sigmas, correls=correls, weights=weights, dtype=np.float64)
            assert round(call, 2) == expected_calls[i]


    test_3spread()
    test_basket()
