import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# reading a csv file using pandas library
wcat=pd.read_csv("C:\\NOTES WRITTEN\Regression\\wc-at.csv")
wcat.columns
wcat.shape

plt.hist(wcat.Waist)
plt.hist(wcat.AT)
plt.boxplot(wcat.Waist,0,"rs",0)
plt.scatter(wcat.Waist,wcat.AT);plt.xlabel("Waist");plt.ylabel("AT")
plt.plot(wcat.Waist,wcat.AT);plt.xlabel("Waist");plt.ylabel("AT")


import matplotlib.pylab as plt
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='red');
plt.plot(wcat['Waist'],pred,color='black');plt.xlabel('WAIST');plt.ylabel('TISSUE')

#FINDING CORRELATION BETWEEN X AND Y IN 2 WAYS
np.corrcoef(wcat.Waist,wcat.AT)
np.corrcoef(wcat.AT,wcat.Waist)

wcat.AT.corr(wcat.Waist)

#LETS BUILD THE MODEL WILL CHECK FOR B0 AND B1 COEFFICIENTS

mod1=smf.ols("AT~Waist",data=wcat).fit()
mod1_summary=mod1.summary()
"""
r=0.81855781
R-squared:                       0.670 it says here if i vary 1% in waist there will 67% variation in AT
Adj. R-squared:                  0.667
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept(B0)   -215.9815     21.796     -9.909      0.000    -259.190    -172.773
Waist(B1)          3.4589      0.235     14.740      0.000       2.994       3.924
==============================================================================

"""
mod1_pred=mod1.predict(wcat)
# LETS FIND THE CORRELATION BETWEEN ACTAL AND PRESICTED VALES
mod1_pred
mod1_errors_Residuals=wcat.AT-mod1_pred
mod1_errors_Residuals
#FINAL CORRELATION WITH ACTUAL AND PREDICTED
mod1_pred.corr(wcat.AT)

""" NOW WE HVE TO CHECK HETEROSCADASTICITY IS DER OR NOT"""
plt.scatter(wcat.Waist,wcat.AT,color="red")
plt.plot(wcat.Waist,mod1_pred);plt.xlabel("Waist");plt.ylabel("AT")

# HERE FINAL PREDICTION IS ALSO 0.8185578128958535 SO LETS GO TO TRANSFORMATION TECHNIQUE

#------------------------------------------------------------------------------------------#
#LOG MODEL
plt.scatter(np.log(wcat.Waist),wcat.AT,color="green")
wcat.AT.corr(np.log(wcat.Waist))

mod2=smf.ols("wcat.AT~np.log(wcat.Waist)",data=wcat).fit()
mod2.params# GIVES B0 AND B1

mod2.summary()
"""
r=0.8217781862645355
R-squared:                       0.675
Adj. R-squared:                  0.672

======================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------
Intercept          -1328.3420     95.923    -13.848      0.000   -1518.498   -1138.186
np.log(wcat.Waist)   317.1356     21.258     14.918      0.000     274.994     359.277
==============================================================================

"""
mod2_pred=mod2.predict(wcat.Waist)
mod2_errors_Residuals=wcat.AT-mod2_pred
#FINAL CORRELATION WITH ACTUAL AND PREDICTED
mod2_pred.corr(wcat.AT)
""" NOW WE HVE TO CHECK HETEROSCADASTICITY IS DER OR NOT"""
plt.scatter(wcat.Waist,wcat.AT,color="red")
plt.plot(wcat.Waist,mod2_pred);plt.xlabel("Waist");plt.ylabel("AT");

#--------------------------------------------------------------------------#


#EXPONENTIAL MODEL

mod3=smf.ols("np.log(AT)~wcat.Waist",data=wcat).fit()
mod3.params
mod3.summary()
print(mod3.conf_int(0.01)) # 99% confidence level
"""
 R-squared:                       0.707
 Adj. R-squared:                  0.704
 ==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.7410      0.233      3.185      0.002       0.280       1.202
wcat.Waist     0.0403      0.003     16.073      0.000       0.035       0.045
==============================================================================
 
"""
mod3_pred=mod3.predict(wcat.Waist)
mod3_errors_Residuals=wcat.AT-mod3_pred
#THERE IS A DIFFERENCE BETWEEN ERRORS SO WE HAVE TO CONVERT TO ANTILOG METHOD THROUGH EXP METHOD

mod3_pred_anti_log=np.exp(mod3_pred)
#FINAL CORRELATION WITH ACTUAL AND PREDICTED
mod3_pred_anti_log.corr(wcat.AT)# 0.0.763380458365053

""" NOW WE HVE TO CHECK HETEROSCADASTICITY IS DER OR NOT"""
plt.scatter(wcat.Waist,wcat.AT,color="red")
plt.plot(wcat.Waist,mod3_pred_anti_log);plt.xlabel("Waist");plt.ylabel("AT");
#--------------------------------------------------------------------------#



#WILL GO TO NON LINEAR 2 DEGREE QUADRATIC EQUATION
wcat["Waist_sq"]=wcat.Waist*wcat.Waist
quad_mod=smf.ols("np.log(wcat.AT)~wcat.Waist+wcat.Waist_sq",data=wcat).fit()
plt.scatter(wcat.Waist_sq,wcat.AT,color="red")
quad_mod.summary()
print(quad_mod.conf_int(0.01)) # 99% confidence level
"""
R-squared:                       0.779
Adj. R-squared:                  0.775
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -7.8241      1.473     -5.312      0.000     -10.744      -4.904
wcat.Waist        0.2289      0.032      7.107      0.000       0.165       0.293
wcat.Waist_sq    -0.0010      0.000     -5.871      0.000      -0.001      -0.001
==============================================================================

"""
quad_mod_pred=quad_mod.predict(wcat)
#THESE VALUES ARE IN LOG BECAUSE np.log(wcat.At) SO WE HAVE TO CONVERT THEM INOT NORMAL THROUGH AMTILOG
quad_mod_anti_log=np.exp(quad_mod_pred)
quad_mod_errors_Residuals=wcat.AT-quad_mod_anti_log
quad_mod_anti_log.corr(wcat.AT)# 0.8285111314182692

""" I WANT TO SEE THE SCATTER DIAGRAM FOR RESIDUAL ERROR TO CHECK THE HOMOSCADASTICITY"""

import statsmodels.api as sm
fig = sm.qqplot(quad_mod_errors_Residuals)
 #plt.show()

""" NOW WE HVE TO CHECK HETEROSCADASTICITY IS DER OR NOT"""
plt.scatter(wcat.Waist,wcat.AT,color="red")
plt.plot(wcat.Waist,quad_mod_anti_log);plt.xlabel("Waist");plt.ylabel("AT")

#---------------------PERFECT MODEL AFTER FINAL CORRELATION ---------#
