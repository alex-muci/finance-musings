"""
TODO: first insert analytical greeks in bs.py & then

    1.  write TEST for Black prices and implieds
     PUT
     f = 101.0
     x = 102.0
     t = .5
     r = .01
     sigma_price = 0.2
     price = black(-1, f, x, t, r=0, sigma_price)
     expected_price = 6.20451158097

     from expected price get implied of 0.2

     CALL
     f= x = 100
     sigma_price = 0.2
     t = .5
     r = .02
     expected_discounted_call_price = 5.5811067246

     implied_vol(cp, f, x, r, t, expected_discounted_call_price)


    2. test analytical vs. numerical here

"""