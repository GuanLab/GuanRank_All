suppressMessages(library("survival"))
suppressMessages(library("timeROC"))
suppressMessages(library("Bolstad2"))
suppressMessages(library("ROCR"))

# question 1A
# riskScoreGlobal is the global risk score
# riskScore12, riskScore18, riskScore24 are the risk scores at 12, 18 and 24 months
# time is called LKADT_P in the CoreTable meaning the last known follow up time in days
# death is last known follow up status (F=survival, T=death)
# all input parameters are vectors
# returned value is a vector containing:
#  * concordance index
#  * AUC of ROC at 12, 18, and 24 months
#  * integrated AUC (integrated over all time points)
## from Justin: https://mail.google.com/mail/u/1/#inbox/14c8f82a26d23b80
score_q1a<-function(time, death, riskScoreGlobal, riskScore12, riskScore18, riskScore24)
{

  if (missing(riskScore12)) {
    riskScore12 <- riskScoreGlobal
  }
  if (missing(riskScore18)) {
    riskScore18 <- riskScoreGlobal
  }
  if (missing(riskScore24)) {
    riskScore24 <- riskScoreGlobal
  }

  auc12 <- timeROC(T=time,
                  delta=death,
                  marker=riskScore12,
                  cause=1,
                  weighting="marginal",
                  times=12 * 30.5,
                  iid=FALSE)$AUC[2]

  auc18 <- timeROC(T=time,
                   delta=death,
                   marker=riskScore18,
                   cause=1,
                   weighting="marginal",
                   times=18 * 30.5,
                   iid=FALSE)$AUC[2]

  auc24 <- timeROC(T=time,
                   delta=death,
                   marker=riskScore24,
                   cause=1,
                   weighting="marginal",
                   times=24 * 30.5,
                   iid=FALSE)$AUC[2]

  # compute global concordance index
  surv <- Surv(time,death)
  cIndex <- survConcordance(surv ~ riskScoreGlobal)$concordance

  # compute iAUC from 6 to 30 months
  times <- seq(6,30,by=1) * 30.5
  aucs <- timeROC(T=time,
                  delta=death,
                  marker=riskScoreGlobal,
                  cause=1,
                  weighting="marginal",
                  times=times,
                  iid=FALSE)$AUC

  # Simpsons rules for integrating under curve
  iAUC <- sintegral(times, aucs)$int / (max(times) - min(times))

  return (list(cIndex=cIndex, auc12=auc12, auc18=auc18, auc24=auc24, iAUC=iAUC))
}

# question 1B
# predTime is the predicted exact survival time for all patients in days
# LKADT_P is last known follow up time in days
# DEATH is last known follow up status (F=survival, T=death)
# all input parameters are vectors
# returned value is RMSE
score_q1b<-function(predTime,LKADT_P,DEATH)
{
  x=LKADT_P[DEATH==T]
  y=predTime[DEATH==T]
  sqrt(sum((x-y)^2)/length(x))
}

# question 2
# pred is the predicted risk with a higher value corresponding to a higher
# probility of discontinue due to AE before 3 months
# y is the censored true label of discontinue due to AE before 3 month
# y=1 if it happens, 0, otherwise
# both input parameters are vectors
# returned value is AUC of PR
score_q2<-function(pred,y)
{
  ## remove subjects whose discontinuation status is NA
  pred <- pred[!is.na(y)]
  y <- y[!is.na(y)]

  prf=ROCR::performance(ROCR::prediction(pred,y),"prec","rec")
  x=prf@x.values[[1]]
  y=prf@y.values[[1]]
  auc=sum((x[-1]-x[-length(x)])*y[-1])
  auc
}

