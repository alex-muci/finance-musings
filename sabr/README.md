Calcolare forward sintetico, per ovviare a:
       (i)  dividendi non osservabili,
       (ii) futures asyncronici (CBOE vs CME) o non esistenti per ETF,
      (iii) liquidita' tra call e put.


Inputs per ogni maturity (una volta al giorno):
   call/put prices,
   one rate (ma non divs),
   spot aiuta ma non necessario.

Updatare durante il giorno solo lo spot:
    New_Fwd = New_Spot * calibrated_Fwd/calibrated_Spot)
    vols = Sticky strike or sticky delta?

Steps:
 1. troviamo call e put intorno allo spot coi prezzi piu simili
    e chiamiamo quello strike K_fwd
 2. Per la call-put parity, il forward e' dato da:
      Fwd_t = K_fwd + [call(K_fwd) - put(K_fwd) ] / (1+r * t)
 3. dato il Fwd_t esprimiamo tutto in funzione delle calls:
      call(k) = put(k) + (Fwd_t - K)* (1+ r * t)
 4. ricaviamo le B&S implied vols (scipy, poi nel futuro Jackel's fast calcs)
 5. fittiamo un polinomio di terzo grado alle 'implied vol' che abbiamo come funzione della moneyness (x := K/Fwd) or del delta, per esempio:
      sigma(x) = a + b * x - c * x^2 + d * x^3
 6. calibriamo Sabr su scadenze liquidi (quelle con piu' di 4 opzioni?)
 7. interpoliamo i parametri di Sabr sulle scadenze meno liquide
 8. calcoliamo prezzi Europee e Americane, relative greche (almeno Delta)
